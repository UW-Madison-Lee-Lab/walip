

imagenet_en_templates = [
    'a bad photo of a {}.',
    'a photo of many {}.',
    'a sculpture of a {}.',
    'a photo of the hard to see {}.',
    'a low resolution photo of the {}.',
    'a rendering of a {}.',
    'graffiti of a {}.',
    'a bad photo of the {}.',
    'a cropped photo of the {}.',
    'a tattoo of a {}.',
    'the embroidered {}.',
    'a photo of a hard to see {}.',
    'a bright photo of a {}.',
    'a photo of a clean {}.',
    'a photo of a dirty {}.',
    'a dark photo of the {}.',
    'a drawing of a {}.',
    'a photo of my {}.',
    'the plastic {}.',
    'a photo of the cool {}.',
    'a close-up photo of a {}.',
    'a black and white photo of the {}.',
    'a painting of the {}.',
    'a painting of a {}.',
    'a pixelated photo of the {}.',
    'a sculpture of the {}.',
    'a bright photo of the {}.',
    'a cropped photo of a {}.',
    'a plastic {}.',
    'a photo of the dirty {}.',
    'a jpeg corrupted photo of a {}.',
    'a blurry photo of the {}.',
    'a photo of the {}.',
    'a good photo of the {}.',
    'a rendering of the {}.',
    'a {} in a video game.',
    'a photo of one {}.',
    'a doodle of a {}.',
    'a close-up photo of the {}.',
    'a photo of a {}.',
    'the origami {}.',
    'the {} in a video game.',
    'a sketch of a {}.',
    'a doodle of the {}.',
    'a origami {}.',
    'a low resolution photo of a {}.',
    'the toy {}.',
    'a rendition of the {}.',
    'a photo of the clean {}.',
    'a photo of a large {}.',
    'a rendition of a {}.',
    'a photo of a nice {}.',
    'a photo of a weird {}.',
    'a blurry photo of a {}.',
    'a cartoon {}.',
    'art of a {}.',
    'a sketch of the {}.',
    'a embroidered {}.',
    'a pixelated photo of a {}.',
    'itap of the {}.',
    'a jpeg corrupted photo of the {}.',
    'a good photo of a {}.',
    'a plushie {}.',
    'a photo of the nice {}.',
    'a photo of the small {}.',
    'a photo of the weird {}.',
    'the cartoon {}.',
    'art of the {}.',
    'a drawing of the {}.',
    'a photo of the large {}.',
    'a black and white photo of a {}.',
    'the plushie {}.',
    'a dark photo of a {}.',
    'itap of a {}.',
    'graffiti of the {}.',
    'a toy {}.',
    'itap of my {}.',
    'a photo of a cool {}.',
    'a photo of a small {}.',
    'a tattoo of the {}.',
]

ru_imagenet_template = [
    'плохая фотография {}.',
    'фото многих {}.',
    'скульптура {}.',
    'фото плохо различимого {}.',
    'фотография с низким разрешением {}.',
    'рендеринг {}.',
    'граффити {}.',
    'плохая фотография {}.',
    'обрезанное фото {}.',
    'тату {}.',
    'вышитый {}.',
    'фото плохо различимого {}.',
    'яркая фотография {}.',
    'фото чистого {}.',
    'фото грязного {}.',
    'темное фото {}.',
    'рисунок {}.',
    'фото моего {}.',
    'пластик {}.',
    'фото крутого {}.',
    'фотография {}.',
    'черно-белая фотография {}.',
    'картина {}.',
    'картина {}.',
    'пиксельная фотография {}.',
    'скульптура {}.',
    'яркая фотография {}.',
    'обрезанное фото {}.',
    'пластик {}.',
    'фото грязного {}.',
    'испорченная фотография {} в формате jpeg.',
    'размытое фото {}.',
    'фото {}.',
    'хорошее фото {}.',
    'рендеринг {}.',
    '{} в видеоигре.',
    'фото одного {}.',
    'каракули {}.',
    'фотография {}.',
    'фото {}.',
    'оригами {}.',
    '{} в видеоигре.',
    'набросок {}.',
    'каракули {}.',
    'оригами {}.',
    'фотография с низким разрешением {}.',
    'игрушка {}.',
    'воспроизведение {}.',
    'фото чистого {}.',
    'фотография большого {}.',
    'воспроизведение {}.',
    'фото приятного {}.',
    'фото странного {}.',
    'размытое фото {}.',
    'мультфильм {}.',
    'искусство {}.',
    'набросок {}.',
    'вышитый {}.',
    'пиксельная фотография {}.',
    'нажмите {}.',
    'испорченная фотография {} в формате jpeg.',
    'хорошее фото {}.',
    'плюшевый {}.',
    'фото красивого {}.',
    'фото маленького {}.',
    'фото странного {}.',
    'мультфильм {}.',
    'искусство {}.',
    'рисунок {}.',
    'фотография большого {}.',
    'черно-белая фотография {}.',
    'плюшевый {}.',
    'темная фотография {}.',
    'нажмите {}.',
    'граффити {}.',
    'игрушка {}.',
    'itap моего {}.',
    'фото классного {}.',
    'фото маленького {}.',
    'тату {}.',
]

prompts = {
    'cifar10': {
        'en': [
            'a photo of a {}.',
            'a blurry photo of a {}.',
            'a black and white photo of a {}.',
            'a low contrast photo of a {}.',
            'a high contrast photo of a {}.',
            'a bad photo of a {}.',
            'a good photo of a {}.',
            'a photo of a small {}.',
            'a photo of a big {}.',
            'a photo of the {}.',
            'a blurry photo of the {}.',
            'a black and white photo of the {}.',
            'a low contrast photo of the {}.',
            'a high contrast photo of the {}.',
            'a bad photo of the {}.',
            'a good photo of the {}.',
            'a photo of the small {}.',
            'a photo of the big {}.',
        ],
        'ru': [
            '{}', 
            'это {}', 
            'на картинке {}', 
            'это {}, домашнее животное'
        ],

        'en2': [
            'a photo of a {}.',
            'a blurry photo of a {}.',
            'a black and white photo of a {}.',
            'a low contrast photo of a {}.',
            'a high contrast photo of a {}.',
            'a bad photo of a {}.',
            'a good photo of a {}.',
            'a photo of a small {}.',
            'a photo of a big {}.',
            'a photo of the {}.',
            'a blurry photo of the {}.',
            'a black and white photo of the {}.',
            'a low contrast photo of the {}.',
            'a high contrast photo of the {}.',
            'a bad photo of the {}.',
            'a good photo of the {}.',
            'a photo of the small {}.',
            'a photo of the big {}.',
        ],
        'it': [
            "una foto di un {}.",
            "una foto di {}",
            "l'immagine di {}",
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
        ],
        'sw' : [
            "Detta är ett foto av en {}.",
            "Detta är ett suddigt foto av en {}.",
            "Detta är ett svartvitt foto av en {}.",
            "ett lågkontrastfoto av en {}.",
            "ett högkontrastfoto av en {}.",
            "ett dåligt foto av en {}.",
            "ett bra foto av en {}.",
            "ett foto på en liten {}.",
            "ett foto av en stor {}.",
            "ett foto av {}.",
            "ett suddigt foto av {}.",
            "ett svartvitt foto av {}.",
            "ett lågkontrastfoto av {}.",
            "ett högkontrastfoto av {}.",
            "ett dåligt foto av {}.",
            "ett bra foto av {}.",
            "ett foto på den lilla {}.",
            "ett foto på den stora {}.",
        ]
    },
    'cifar100': {
        'en': [
            'a photo of a {}.',
            'a blurry photo of a {}.',
            'a black and white photo of a {}.',
            'a low contrast photo of a {}.',
            'a high contrast photo of a {}.',
            'a bad photo of a {}.',
            'a good photo of a {}.',
            'a photo of a small {}.',
            'a photo of a big {}.',
            'a photo of the {}.',
            'a blurry photo of the {}.',
            'a black and white photo of the {}.',
            'a low contrast photo of the {}.',
            'a high contrast photo of the {}.',
            'a bad photo of the {}.',
            'a good photo of the {}.',
            'a photo of the small {}.',
            'a photo of the big {}.',
        ],
         'ru1': [
            '{}', 
            'это {}', 
            'на картинке {}', 
            'это {}, домашнее животное'
        ],
        'ru': [
            'фото {}.',
             'размытое фото {}.',
             'черно-белая фотография {}.',
             'низкоконтрастное фото {}.',
             'высококонтрастное фото {}.',
             'плохая фотография {}.',
             'хорошее фото {}.',
             'фото маленького {}.',
             'фотография большого {}.',
             'фото {}.',
             'размытое фото {}.',
             'черно-белая фотография {}.',
             'низкоконтрастное фото {}.',
             'высококонтрастное фото {}.',
             'плохая фотография {}.',
             'хорошее фото {}.',
             'фото маленького {}.',
             'фото большого {}.',
        ],
        'en2': [
            'a photo of a {}.',
            'a blurry photo of a {}.',
            'a black and white photo of a {}.',
            'a low contrast photo of a {}.',
            'a high contrast photo of a {}.',
            'a bad photo of a {}.',
            'a good photo of a {}.',
            'a photo of a small {}.',
            'a photo of a big {}.',
            'a photo of the {}.',
            'a blurry photo of the {}.',
            'a black and white photo of the {}.',
            'a low contrast photo of the {}.',
            'a high contrast photo of the {}.',
            'a bad photo of the {}.',
            'a good photo of the {}.',
            'a photo of the small {}.',
            'a photo of the big {}.',
        ],
        'it': [
            "una foto di un {}.",
            "una foto di {}",
            "l'immagine di {}",
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
        ],
        'sw' : [
            "Detta är ett foto av en {}.",
            "Detta är ett suddigt foto av en {}.",
            "Detta är ett svartvitt foto av en {}.",
            "ett lågkontrastfoto av en {}.",
            "ett högkontrastfoto av en {}.",
            "ett dåligt foto av en {}.",
            "ett bra foto av en {}.",
            "ett foto på en liten {}.",
            "ett foto av en stor {}.",
            "ett foto av {}.",
            "ett suddigt foto av {}.",
            "ett svartvitt foto av {}.",
            "ett lågkontrastfoto av {}.",
            "ett högkontrastfoto av {}.",
            "ett dåligt foto av {}.",
            "ett bra foto av {}.",
            "ett foto på den lilla {}.",
            "ett foto på den stora {}.",
        ]
    },
    'imagenet': {
        'en': [
            '{}',
             'this {}',
             'on the picture {}',
             'this is {}, pet'
        ],
        'ru': [
            '{}', 
            'это {}', 
            'на картинке {}', 
            'это {}, домашнее животное'
        ],
        'en1': [
            "itap of a {}.",
            "a bad photo of the {}.",
            "a origami {}.",
            "a photo of the large {}.",
            "a {} in a video game.",
            "art of the {}.",
            "a photo of the small {}."
        ],
        'en2': [
            "itap of a {}.",
            "a bad photo of the {}.",
            "a origami {}.",
            "a photo of the large {}.",
            "a {} in a video game.",
            "art of the {}.",
            "a photo of the small {}."
        ],
        'ru1': [
            "нажатие {}.",
            "плохая фотография {}.",
            "оригами {}.",
            "фото большого {}.",
            "{} в видеоигре.",
            "искусство {}.",
            "фото маленького {}."
        ],
        'it': [
            "itap di un {}.",
            "una brutta foto del {}.",
            "un origami {}.",
            "una foto del grande {}.",
            "a {} in un videogioco.",
            "arte del {}.",
            "una foto del piccolo {}."
        ],
        'sw': [
            "tap på en {}.",
            "ett dåligt foto av {}.",
            "en origami {}.",
            "ett foto av den stora {}.",
            "en {} i ett videospel.",
            "konst av {}.",
            "ett foto på den lilla {}."
        ]
    },
    'imagenet_r': {
        'en': [
            "itap of a {}.",
            "a bad photo of the {}.",
            "a origami {}.",
            "a photo of the large {}.",
            "a {} in a video game.",
            "art of the {}.",
            "a photo of the small {}."
        ],
        'it': [
            "itap di un {}.",
            "una brutta foto del {}.",
            "un origami {}.",
            "una foto del grande {}.",
            "a {} in un videogioco.",
            "arte del {}.",
            "una foto del piccolo {}."
        ]
    }
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

