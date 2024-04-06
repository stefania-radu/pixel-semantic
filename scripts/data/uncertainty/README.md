# Get test data in dictionary form for each task:
- NER: `python scripts\data\uncertainty\gather_test_data_ner.py 10`
- Tydiqa: `python scripts\data\uncertainty\gather_test_data_tydiqa.py 10`
- GLUE: `python scripts\data\uncertainty\gather_test_data_glue.py 10`

Here N=10 examples per language for NER, Tydiqa and 10 examples per subtask for GLUE

Data has this format:

```json
{
    "amh": {
        "amh_0": "በምግብ\nላይ\nጨው\nማብዛትና\nየደም\nጋፊት\nማየልም\nዕድሜን\nእንደሚያሳጥሩ\nየሚያጠራጥር\nአልሆነም\n።",
        "amh_1": "የጥረቱ\nምንጭ\nበእርግጥ\nዘርፈ\nብዙ\nእና\nሰፊ\nነው\nከማጭበርበር\nአንስቶ\nየፖለቲካ\nተልዕኮ\nእስከ\nማስፈጸም\nሊደርስ\nይችላል\n።",
        "amh_2": "በዚህ\nሳምንት\nበማኅበራዊ\nመገናኛ\nዘዴዎች\nመነጋገሪያ\nከኾኑ\nነጥቦች\nመካከል\nአራት\nጉዳዮች\nጎልተው\nወጥተዋል\n።",
        "amh_3": "ከተለያዩ\nየማኅበረሰቡ\nአባላት\nጋር\nከጀመሩት\nንግግር\nባሻገር\nሌሎች\nጉዳዮችን\nለመመልከት\nጊዜ\nያስፈልጋቸዋል\nሲሉ\nየሚሞግቱም\nአሉ\n።",
        "amh_4": "የዚህ\nሁሉ\nመንስኤ\nያሁኑ\nመንግስት\nስልጣን\nከያዘ\nአንስቶ\nያለዉ\nየመልካም\nአስተዳደር\nብልሹነት\nነዉ\nሲሉ\nአቶ\nኝካዉ\nኦቻላ\nያስረዳሉ\n።",
        "amh_5": "ሞት\nይሻላል\nበዘመኔ\nይሄን\nዝቅጠት\nከማይ\nየኤሊያስ\nከበደ\nጣፋ\nየፌስቡክ\nመልእክት\nነው\n።",
        "amh_6": "አዛውንቶችን\nጧሪ\nሳይሆን\nደብዳቢ\nወጣቶችን\nኢትዮጵያ\nማፍራቷ\nነው\nልብ\nየሚሰብረው\nየሚል\nመልእክት\nትዊተር\nገጿ\nላይ\nአስፍራለች\n።",
        "amh_7": "ሆኖም\nለየት\nየሚያደርጋቸው\nየአፕሊኬሽኖቹ\nባለቤቶች\nያልተቆለፈው\nመልእክት\nየሰነድ\nማከማቻ\nቋታቸ\nውስጥ\nመገኘቱ\nነው\n።",
        "amh_8": "የማኅበራዊ\nመገናኛ\nዘዴዎች\nቅኝት\nጠቅላይ\nሚንሥትር\nአቢይ\nአህመድ\nየአምቦ\nጉብኝት\nእና\nየኅብረተሰቡ\nየለውጥ\nተማጽኖን\nይዳስሳል\n።",
        "amh_9": "የኮምፒውተሮች\nመስፋፋት\nእና\nመርቀቅ\nግን\nየተቆለፈው\nመልእክት\nእጅግ\nውስብስብ\nእንዲሆን\nእና\nበቀላሉ\nእንዳይፈታ\nለማድረግ\nአግዘዋል\n።"
    },
```

# Combine all data into one

`python scripts\data\uncertainty\combine_test_data.py`

Data has this format:

```json
{
    "ner": {
        "amh": {
            "amh_0": "በምግብ\nላይ\nጨው\nማብዛትና\nየደም\nጋፊት\nማየልም\nዕድሜን\nእንደሚያሳጥሩ\nየሚያጠራጥር\nአልሆነም\n።",
            "amh_1": "የጥረቱ\nምንጭ\nበእርግጥ\nዘርፈ\nብዙ\nእና\nሰፊ\nነው\nከማጭበርበር\nአንስቶ\nየፖለቲካ\nተልዕኮ\nእስከ\nማስፈጸም\nሊደርስ\nይችላል\n።",
            "amh_2": "በዚህ\nሳምንት\nበማኅበራዊ\nመገናኛ\nዘዴዎች\nመነጋገሪያ\nከኾኑ\nነጥቦች\nመካከል\nአራት\nጉዳዮች\nጎልተው\nወጥተዋል\n።",
            "amh_3": "ከተለያዩ\nየማኅበረሰቡ\nአባላት\nጋር\nከጀመሩት\nንግግር\nባሻገር\nሌሎች\nጉዳዮችን\nለመመልከት\nጊዜ\nያስፈልጋቸዋል\nሲሉ\nየሚሞግቱም\nአሉ\n።",
            "amh_4": "የዚህ\nሁሉ\nመንስኤ\nያሁኑ\nመንግስት\nስልጣን\nከያዘ\nአንስቶ\nያለዉ\nየመልካም\nአስተዳደር\nብልሹነት\nነዉ\nሲሉ\nአቶ\nኝካዉ\nኦቻላ\nያስረዳሉ\n።",
            "amh_5": "ሞት\nይሻላል\nበዘመኔ\nይሄን\nዝቅጠት\nከማይ\nየኤሊያስ\nከበደ\nጣፋ\nየፌስቡክ\nመልእክት\nነው\n።",
            "amh_6": "አዛውንቶችን\nጧሪ\nሳይሆን\nደብዳቢ\nወጣቶችን\nኢትዮጵያ\nማፍራቷ\nነው\nልብ\nየሚሰብረው\nየሚል\nመልእክት\nትዊተር\nገጿ\nላይ\nአስፍራለች\n።",
            "amh_7": "ሆኖም\nለየት\nየሚያደርጋቸው\nየአፕሊኬሽኖቹ\nባለቤቶች\nያልተቆለፈው\nመልእክት\nየሰነድ\nማከማቻ\nቋታቸ\nውስጥ\nመገኘቱ\nነው\n።",
            "amh_8": "የማኅበራዊ\nመገናኛ\nዘዴዎች\nቅኝት\nጠቅላይ\nሚንሥትር\nአቢይ\nአህመድ\nየአምቦ\nጉብኝት\nእና\nየኅብረተሰቡ\nየለውጥ\nተማጽኖን\nይዳስሳል\n።",
            "amh_9": "የኮምፒውተሮች\nመስፋፋት\nእና\nመርቀቅ\nግን\nየተቆለፈው\nመልእክት\nእጅግ\nውስብስብ\nእንዲሆን\nእና\nበቀላሉ\nእንዳይፈታ\nለማድረግ\nአግዘዋል\n።"
        },
        "conll_2003_en": {
            "conll_2003_en_0": "SOCCER\n-\nLEADING\nSCOTTISH\nPREMIER\nDIVISION\nSCORERS\n.",
            "conll_2003_en_1": "Torquay\n22\n8\n4\n10\n22\n24\n28",
            "conll_2003_en_2": "New\nZealand\nPrime\nMinister\nJim\nBolger\n,\nemerging\nfrom\ncoalition\ntalks\nwith\nthe\nnationalist\nNew\nZealand\nFirst\nparty\non\nFriday\nafternoon\n,\nsaid\nNational\nand\nNZ\nFirst\nwould\nmeet\nagain\non\nSunday\n.",
            "conll_2003_en_3": "League\nteams\nafter\ngames\nplayed\non\nThursday\n(\ntabulate\nunder",
            "conll_2003_en_4": "21.\nShannon\nNobis\n(\nU.S.\n)\n1:19.08",
            "conll_2003_en_5": "Canola\n47.1\n988.1\n1135.5\n46.7\n894.9\n822.0",
            "conll_2003_en_6": "8-213\n9-216\n.",
            "conll_2003_en_7": "C.\nHarris\nlbw\nb\nWasim\n22",
            "conll_2003_en_8": "TORONTO\n11\n15\n0\n76\n89\n22",
            "conll_2003_en_9": "with\nthe\nUnited\nStates\nand\nwith\nthe\nfirst\nwoman\never\nto\nhold\nthe\nposition\nof\nSecretary\nof\nState\n."
        },
```


# For 1000 elements:

## NER
Counts: {'amh': 500, 'conll_2003_en': 1000, 'hau': 552, 'ibo': 638, 'kin': 605, 'lug': 407, 'luo': 186, 'pcm': 600, 'swa': 604, 'wol': 539, 'yor': 645, 'zh': 1000}

## Tydiqa
Counts: {'arabic': 921, 'russian': 812, 'bengali': 113, 'telugu': 669, 'finnish': 782, 'swahili': 499, 'korean': 276, 'indonesian': 565, 'english': 440}

## GLUE
Counts: {'cola': 1000, 'mnli': 1000, 'mrpc': 1000, 'qnli': 1000, 'qqp': 1000, 'rte': 1000, 'sst2': 1000, 'stsb': 1000, 'wnli': 146}