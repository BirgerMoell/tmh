from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import MarianMTModel, MarianTokenizer
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
ARTICLE = """ New York (CNN)When Liana Barrientos was 23 years old, she got married in Westchester County, New York.
 A year later, she got married again in Westchester County, but to a different man and without divorcing her first husband.
 Only 18 days after that marriage, she got hitched yet again. Then, Barrientos declared "I do" five more times, sometimes only within two weeks of each other.
 In 2010, she married once more, this time in the Bronx. In an application for a marriage license, she stated it was her "first and only" marriage.
 Barrientos, now 39, is facing two criminal counts of "offering a false instrument for filing in the first degree," referring to her false statements on the
 2010 marriage license application, according to court documents.
 Prosecutors said the marriages were part of an immigration scam.
 On Friday, she pleaded not guilty at State Supreme Court in the Bronx, according to her attorney, Christopher Wright, who declined to comment further.
 After leaving court, Barrientos was arrested and charged with theft of service and criminal trespass for allegedly sneaking into the New York subway through an emergency exit, said Detective
 Annette Markowski, a police spokeswoman. In total, Barrientos has been married 10 times, with nine of her marriages occurring between 1999 and 2002.
 All occurred either in Westchester County, Long Island, New Jersey or the Bronx. She is believed to still be married to four men, and at one time, she was married to eight men at once, prosecutors say.
 Prosecutors said the immigration scam involved some of her husbands, who filed for permanent residence status shortly after the marriages.
 Any divorces happened only after such filings were approved. It was unclear whether any of the men will be prosecuted.
 The case was referred to the Bronx District Attorney\'s Office by Immigration and Customs Enforcement and the Department of Homeland Security\'s
 Investigation Division. Seven of the men are from so-called "red-flagged" countries, including Egypt, Turkey, Georgia, Pakistan and Mali.
 Her eighth husband, Rashid Rajput, was deported in 2006 to his native Pakistan after an investigation by the Joint Terrorism Task Force.
 If convicted, Barrientos faces up to four years in prison.  Her next court appearance is scheduled for May 18.
 """

# print(summarizer(ARTICLE, max_length=130, min_length=30, do_sample=False))

def get_summary(text, model="t5-base"):
    summarizer = pipeline("summarization")
    result =summarizer(text, max_length=130, min_length=30, do_sample=False)
    return result


def translate_between_languages(text, model):

    tokenizer = AutoTokenizer.from_pretrained(model)
    model = AutoModelForSeq2SeqLM.from_pretrained(model)

    translated = model.generate(**tokenizer(text, return_tensors="pt", padding=True))
    output = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]
    # print(output)
    return output[0]

def pegasus_summary(text):
  
    tokenizer = AutoTokenizer.from_pretrained("google/pegasus-xsum")

    model = AutoModelForSeq2SeqLM.from_pretrained("google/pegasus-xsum")

    batch = tokenizer(text, truncation=True, padding='longest', return_tensors="pt")
    translated = model.generate(**batch)
    tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
    return tgt_text[0]

def translate_and_summarize(swedish_long_text):

    english_long_text = translate_between_languages(swedish_long_text, "Helsinki-NLP/opus-mt-sv-en")
    # print("the long english text is", english_long_text)

    english_summary = pegasus_summary(english_long_text)
    # print("the english summary is", english_summary)

    swedish_summary = translate_between_languages(english_summary, "Helsinki-NLP/opus-mt-en-sv")
    # print("the swedish summary is", swedish_summary)
    return swedish_summary

   

# summary = pegasus_summary(ARTICLE)
# print(summary)
# sv_text = "Albert Einstein var son till Hermann och Pauline Einstein, vilka var icke-religiösa judar och tillhörde medelklassen. Fadern var försäljare och drev senare en elektroteknisk fabrik. Familjen bosatte sig 1880 i München där Einstein gick i en katolsk skola. Mängder av myter cirkulerar om Albert Einsteins person. En av dessa är att han som barn skulle ha haft svårigheter med matematik, vilket dock motsägs av hans utmärkta betyg i ämnet.[15] Han nämnde ofta senare att han inte trivdes i skolan på grund av dess pedagogik. Att Albert Einstein skulle vara släkt med musikvetaren Alfred Einstein är ett, ofta framfört, obevisat påstående. Alfred Einsteins dotter Eva har framhållit att något sådant släktskap inte existerar."

# swedish_summary = translate_and_summarize(sv_text)
# print(swedish_summary)
# sum = get_summary(ARTICLE)
# print(sum)