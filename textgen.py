def top_k_logits(logits, k):
    """
    Masks everything but the k top entries as -infinity (-1e10).
    Used to mask logits such that e^-infinity -> 0 won't contribute to the
    sum of the denominator.
    """
    if k == 0:
        return logits
    else:
        values = torch.topk(logits, k)[0]
        batch_mins = values[:, -1].view(-1, 1).expand_as(logits)
        return torch.where(logits < batch_mins, torch.ones_like(logits) * -1e10, logits)

    
def top_p_logits(logits, p):
    """
    Masks everything but the top-p entries as -infinity (-1e10).
    
    Differently from `top_k_logits`, here we we don't take a fixed number
    k of elements in `logits`, but a fraction `p`
    of elements. These are the elements higher that the `p` percentile.
    
    Used to mask logits such that e^-infinity -> 0 won't contribute to the
    sum of the denominator.
    """
    if p == 1:
        return logits
    else:
        k = round(logits.shape[1] * p)
        return top_k_logits(logits, k)

    
def sample_sequence(model, length, context, batch_size=None, 
                    temperature=1, top_k=0, top_p=None, device='cuda', sample=True):
    print(f'context[{type(context)}]: {context}')
    if not sample:
        assert top_k > 0, f'top_k ({top_k}) needs to be >0 when sample=False'
    context = torch.tensor(context, device=device, dtype=torch.long).unsqueeze(0).repeat(batch_size, 1)
    prev = context.clone()
    output =  context.clone()
    past = None
    with torch.no_grad():
        for i in trange(length):
            # predict next token
            #print(f'context.shape = {context.shape}  past: {None if past is None else [t.shape for t in past]}')
            logits, past = model(prev, past=past)
            vocab_size = logits.shape[-1]
            assert logits.shape == (batch_size, prev.shape[1], vocab_size)
            #print(f'logits.shape = {logits.shape}')
            last_logits = logits[:, -1, :].clone() / temperature
            #print(f'last_logits.shape = {last_logits.shape}')
            # set part of the logits to 0, according to top_k or top_p schemes
            if top_p is None:
                last_logits = top_k_logits(last_logits, k=top_k)
            else:
                last_logits = top_p_logits(last_logits, p=top_p)
            log_probs = F.softmax(last_logits, dim=-1)
            if sample:
                predicted_index = torch.multinomial(log_probs, num_samples=1)
            else:
                #print(f'END[topk] logits.shape = {logits.shape}', flush=True)
                topk = torch.topk(last_logits, k=top_k, dim=-1)
                predicted_index = topk.indices[..., top_k - 1:top_k]
                assert predicted_index == topk.indices[0, top_k - 1]
            prev = predicted_index
            output = torch.cat((output, prev), dim=1)
    return output


def encode_transformer_xl(text, encoder, device):
    text_tokenized = encoder.tokenize(text)
    text_indexed = encoder.convert_tokens_to_ids(text_tokenized)
    #text_indexed_tensor = torch.tensor([text_indexed])
    #text_indexed_tensor = text_indexed_tensor.to(device)
    return text_indexed


def encode_gpt2(text, encoder, device=None):
    return encoder.encode(text)


def generate_text(
        # model specific arguments
        model, 
        encoder_func,
        encoder,
        decoder,
        EOT = '<|endoftext|>',
        # 
        prompt = None,
        batch_size = 1,
        nsamples = 1,    
        length = -1,
        temperature = 1,
        top_k = 0,
        top_p = None,
        sample = True,
        seed = 0,
    ):
    # Arguments checks
    assert nsamples % batch_size == 0
    assert prompt is not None and len(prompt) > 0
    
    # Seed the random-number generators
    if seed is not None:
        np.random.seed(seed)
        torch.random.manual_seed(seed)
        torch.cuda.manual_seed(seed)
    
    # Encode prompt: str -> tokens -> tensor(vocabulary)
    context_tokens = encoder_func(prompt, encoder, device)
    
    # Generate an output text (multiple time if (nsamples / batch_size) > 1)
    generated = 0
    for _ in range(nsamples // batch_size):
        out = sample_sequence(
            model=model, length=length,
            context=context_tokens,
            batch_size=batch_size,
            temperature=temperature, top_k=top_k, top_p=top_p, device=device, sample=sample,
        )
        print(f'PROMPT: {prompt}')
        out = out[:, len(context_tokens):].tolist()
        for i in range(batch_size):
            generated += 1
            text = decoder(out[i])
            print("=" * 40 + " SAMPLE " + str(generated) + " " + "=" * 40)
            end = text.find(EOT)
            end = len(text) if end == -1 else end+len(EOT)
            print(text[:end])
    print("=" * 80)


def decoder_gpt2(ids, enc):
    return enc.decode(ids)


def decoder_transformer_xl(ids, tokenizer):
    return ' '.join([tokenizer.convert_ids_to_tokens([index])[0]
                     for index in ids])


def generate_text_gpt2(
        model, 
        encoder,
        decoder,
        prompt,
        batch_size = 1,
        nsamples = 1,    
        length = -1,
        temperature = 1,
        top_k = 0,
        top_p = None,
        sample = True,
        seed = 0,
    ):
    if length == -1:
        length = model.config.n_ctx // 2
    elif length > model.config.n_ctx:
        raise ValueError("Can't get samples longer than window size: %s" % model.config.n_ctx)
    generate_text(model, encoder_func=encode_gpt2, encoder=encoder, decoder=decoder, EOT='<|endoftext|>',
                prompt = prompt,
                batch_size = batch_size,
                nsamples = nsamples,    
                length = length,
                temperature = temperature,
                top_k = top_k,
                top_p = top_p,
                sample = sample,
                seed = seed)


def generate_text_transformer_xl(
        model,
        encoder,
        decoder,
        prompt,
        batch_size = 1,
        nsamples = 1,    
        length = 32,
        temperature = 1,
        top_k = 0,
        top_p=None,
        sample = True,
        seed = 0,
    ):
    generate_text(model, encoder_func=encode_transformer_xl, encoder=encoder, decoder=decoder, EOT='<eos>',
                prompt = prompt,
                batch_size = batch_size,
                nsamples = nsamples,    
                length = length,
                temperature = temperature,
                top_k = top_k,
                top_p = top_p,
                sample = sample,
                seed = seed)
