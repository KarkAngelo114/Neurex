const tokenize = (sentence) => {
    return sentence
        .toLowerCase()
        .match(/[a-z]+|\d+|[^\w\s]/g) || [];
};

exports.buildVocab = (sentences) => {
    const data = Array.isArray(sentences)
        ? sentences
        : [sentences];

    return [...new Set(data.flatMap(tokenize))];
};


exports.buildWord2Id = (vocab) => {
    const SPECIAL_TOKENS = ["<PAD>", "<UNK>"];
    return Object.fromEntries(
        [...SPECIAL_TOKENS, ...vocab].map((word, i) => [word, i])
    );
};

// Encode now just uses the pre-built word2id
exports.Encode = (sentence, word2id, max_length) => {

    let token_array = tokenize(sentence).map(word => word2id[word] ?? word2id["<UNK>"]);

    // sentences has different lengths, so we either pad (append 0s) or truncate the encoded input sentence
    
    // this block ensures to pad the input if it's less than the given max length
    if (token_array.length < max_length) {
        const needed_length = max_length - token_array.length;
        const pad = new Array(needed_length).fill(0);
        token_array = [...token_array, ...pad];
    }
    // this block is for truncating if the tokenize input is longer than the set maximumn length
    else if (token_array.length > max_length) {
        token_array.length = max_length;
    }

    return token_array;
};