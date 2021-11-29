
eng_templates = [
    "a photo of a {}.",
    "a blurry photo of a {}.",
    "a black and white photo of a {}.",
    "a low contrast photo of a {}.",
    "a high contrast photo of a {}.",
    "a bad photo of a {}.",
    "a good photo of a {}.",
    "a photo of a small {}.",
    "a photo of a big {}.",
    "a photo of the {}.",
    "a blurry photo of the {}.",
    "a black and white photo of the {}.",
    "a low contrast photo of the {}.",
    "a high contrast photo of the {}.",
    "a bad photo of the {}.",
    "a good photo of the {}.",
    "a photo of the small {}.",
    "a photo of the big {}.",
]

ita_templates = [
    "una foto di un {}.",
    "una foto sfocata di un {}.",
    "una foto in bianco e nero di {}.",
    "una foto a basso contrasto di un {}.",
    "una foto ad alto contrasto di un {}.",
    "una brutta foto di un {}.",
    "una buona foto di un {}.",
    "una foto di un piccolo {}.",
    "una foto di un grande {}.",
    "una foto del {}.",
    "una foto sfocata di {}.",
    "una foto in bianco e nero del {}.",
    "una foto a basso contrasto di {}.",
    "una foto ad alto contrasto di {}.",
    "una brutta foto del {}.",
    "una buona foto del {}.",
    "una foto del piccolo {}.",
    "una foto del grande {}.",
]

fre_templates = [
    "une photo d'un {}.",
     "une photo floue d'un {}.",
     "une photo en noir et blanc d'un {}.",
     "une photo à faible contraste d'un {}.",
     "une photo à contraste élevé d'un {}.",
     "une mauvaise photo d'un {}.",
     "une bonne photo d'un {}.",
     "une photo d'un petit {}.",
     "une photo d'un grand {}.",
     "une photo du {}.",
     "une photo floue du {}.",
     "une photo en noir et blanc du {}.",
     "une photo à faible contraste du {}.",
     "une photo à contraste élevé du {}.",
     "une mauvaise photo du {}.",
     "une bonne photo du {}.",
     "une photo du petit {}.",
     "une photo du grand {}.",
]
templates = {
    'en' : eng_templates,
    'it': ita_templates,
    'fr': fre_templates
}

def generate_texts(template, vocabs, k=-1):
    texts = []
    for desc in vocabs:
        if k == -1:
            t = [template[i].format(desc) for i in range(len(template))]
        else:
            t = [template[i].format(desc) for i in range(k)]
        texts += t
    return texts