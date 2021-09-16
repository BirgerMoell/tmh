from transformers import AutoTokenizer, AutoModelWithLMHead, AutoModelForSeq2SeqLM, TranslationPipeline, pipelines

def translate_text(text, model="SEBIS/legal_t5_small_trans_sv_en_small_finetuned", tokenizer="SEBIS/legal_t5_small_trans_sv_en_small_finetuned"):
    pipeline = TranslationPipeline(
    model=AutoModelWithLMHead.from_pretrained(model),
    tokenizer=AutoTokenizer.from_pretrained(pretrained_model_name_or_path = tokenizer, do_lower_case=False, 
                                                skip_special_tokens=True))
    translation = pipeline(text, max_length=512)
    return translation

def translate_between_languages(text, model):

    tokenizer = AutoTokenizer.from_pretrained(model)
    model = AutoModelForSeq2SeqLM.from_pretrained(model)

    translated = model.generate(**tokenizer(text, return_tensors="pt", padding=True))
    output = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]
    # print(output)
    return output[0]


# sv_text = "Albert Einstein var son till Hermann och Pauline Einstein, vilka var icke-religiösa judar och tillhörde medelklassen. Fadern var försäljare och drev senare en elektroteknisk fabrik. Familjen bosatte sig 1880 i München där Einstein gick i en katolsk skola. Mängder av myter cirkulerar om Albert Einsteins person. En av dessa är att han som barn skulle ha haft svårigheter med matematik, vilket dock motsägs av hans utmärkta betyg i ämnet.[15] Han nämnde ofta senare att han inte trivdes i skolan på grund av dess pedagogik. Att Albert Einstein skulle vara släkt med musikvetaren Alfred Einstein är ett, ofta framfört, obevisat påstående. Alfred Einsteins dotter Eva har framhållit att något sådant släktskap inte existerar."

# translation = translate_text(sv_text)
# print(translation)