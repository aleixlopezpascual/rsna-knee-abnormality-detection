from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence

import pandas as pd

from src.config import TARGET_COLS

TARGETS: list[str] = list(TARGET_COLS)

# Turkish dotted/dotless i must be folded before casefolding.
_PRE = str.maketrans({
    "ı": "i",
    "İ": "i",
    "I": "i",
    "ß": "ss",
    "đ": "d",
    "Đ": "d",
    "ø": "o",
    "Ø": "o",
    "æ": "ae",
    "Æ": "ae",
})


def _rx(*alts: str) -> re.Pattern[str]:
    return re.compile("|".join(alts))


def normalize(text: str) -> str:
    """Fold case, diacritics and separators; keep Greek and Cyrillic letters."""
    if not isinstance(text, str):
        return ""
    text = text.translate(_PRE).lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.replace("­", "")  # soft hyphen
    text = re.sub(r"[_\-/\\]+", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text


_SENT_SPLIT = re.compile(r"(?<=[.;!?])\s+|\n+")


def clauses(text: str) -> list[str]:
    """Split into clauses, then attach header: lines to following text."""
    norm = normalize(text)
    raw = [c.strip() for c in _SENT_SPLIT.split(norm) if c and c.strip()]

    merged: list[str] = []
    for i, c in enumerate(raw):
        if c.endswith(":") and len(c.split()) <= 14 and i + 1 < len(raw):
            merged.append(c + " " + raw[i + 1])
        else:
            merged.append(c)

    out: list[str] = []
    for c in merged:
        out.append(c)
        if len(c.split()) > 25:
            out.extend(p.strip() for p in c.split(",") if len(p.split()) > 2)
    return out


NEGATION = _rx(
    r"\bno\b",
    r"\bnot\b",
    r"\bwithout\b",
    r"\bnegative for\b",
    r"\babsence\b",
    r"\bno evidence\b",
    r"\bunremarkable\b",
    r"\bfree of\b",
    r"\bnone\b",
    r"\bnil\b",
    r"\bsin\b",
    r"\bno hay\b",
    r"\bausencia\b",
    r"\bausentes?\b",
    r"\bpas de\b",
    r"\bsans\b",
    r"\baucune?\b",
    r"\bgeen\b",
    r"\bzonder\b",
    r"\bniet\b",
    r"\bkeine?\b",
    r"\bohne\b",
    r"\bnicht\b",
    r"\byok\b",
    r"\byoktur\b",
    r"izlenmemekte",
    r"saptanmadi",
    r"\bdegil\b",
    r"gozlenmemekte",
    r"mevcut degil",
    r"eslik etmiyor",
    r"\bizlenmedi\b",
    r"\bnema\b",
    r"\bbez\b",
    r"\bnisu\b",
    r"\bnije\b",
    r"\bδεν\b",
    r"\bχωρις\b",
    r"ουδεν",
    r"\bбез\b",
    r"\bне\b",
    r"липсва",
    r"\bняма\b",
)

NORMALITY = _rx(
    r"\bnormal",
    r"\bintact\b",
    r"\bpreserved\b",
    r"\bwithin normal limits\b",
    r"limites normales",
    r"\bconservad",
    r"\bintegr",
    r"\bnormales\b",
    r"\bdoga(l|ll)\b",
    r"korunmus",
    r"\bnormaldir\b",
    r"olagan",
    r"\buredn",
    r"\bocuvan",
    r"\bodrzan",
    r"\bintakt",
    r"φυσιολογικ",
    r"ακεραι",
    r"unauffallig",
    r"regelrecht",
    r"\bintakt\b",
    r"нормал",
    r"запазен",
    r"съхранен",
    r"\bбез особености\b",
    r"\bgaaf\b",
    r"\bnormaal\b",
)

UNCERTAIN = _rx(
    r"\bpossible\b",
    r"\bprobable\b",
    r"\bsuspicious\b",
    r"\bsuspected\b",
    r"cannot (be )?exclude",
    r"\bmay\b",
    r"\bquestionable\b",
    r"\bequivocal\b",
    r"\bposible\b",
    r"sin criterios categoricos",
    r"\bdudos",
    r"\bmuhtemel\b",
    r"\bolasi\b",
    r"\bsupheli\b",
    r"\bizlenim",
    r"\bmoguce\b",
    r"\bvjerojatno\b",
    r"\bsumnja\b",
    r"πιθαν",
    r"υποπτ",
    r"\bmoglich",
    r"\bverdachtig",
    r"\bfraglich",
    r"\bV\.a\.\b",
    r"\bвъзможно\b",
    r"\bвероятно\b",
    r"суспект",
    r"\bmogelijk\b",
    r"\bverdacht\b",
)

TEAR = _rx(
    r"\btear",
    r"\btorn\b",
    r"\brupture",
    r"\bdisruption\b",
    r"discontinuit",
    r"\bavuls",
    r"\brotura\b",
    r"\broturas\b",
    r"\bruptura",
    r"\bdesgarro",
    r"\broto\b",
    r"\bdechirure",
    r"\bdechire",
    r"\bscheur",
    r"\bruptuur",
    r"gescheurd",
    r"riss(bildung|e|es)?\b",
    r"einriss",
    r"\bruptur",
    r"zerreiss",
    r"\blasion",
    r"\byirtik",
    r"\byirtig",
    r"\bkopma\b",
    r"butunluk kaybi",
    r"\brupturu\b",
    r"\bpuknuce",
    r"\bruptur",
    r"\bprekid\b",
    r"\bpukotin",
    r"ρηξη",
    r"ρηξις",
    r"ρηγμα",
    r"руптура",
    r"разкъсв",
    r"разрив",
    r"скъсв",
)

DEGEN = _rx(
    r"degenerat",
    r"\bmucoid\b",
    r"\bmyxoid\b",
    r"\bfray",
    r"\bfissur",
    r"dejeneratif",
    r"\bmukoid\b",
    r"degenerativn",
    r"εκφυλ",
    r"дегенерат",
    r"\bμυξοειδ",
    r"\bμυξωδ",
    r"\bmuco ?ide\b",
    r"aufgefasert",
)

INJURY = _rx(
    r"\binjur",
    r"\bsprain",
    r"\blesion",
    r"\blasion",
    r"\bedema\b",
    r"\boedema\b",
    r"\bodem\b",
    r"\bedem\b",
    r"\bοιδημα",
    r"\bодем",
    r"\bедем",
    r"\bstrain\b",
    r"\bhigh signal\b",
    r"\bsignal alteration\b",
    r"\bhiperintens",
    r"\bhyperintens",
    r"aumento de senal",
    r"alteracion de senal",
    r"cambio de senal",
    r"\bsignalanhebung",
    r"\bsignalalteration",
    r"verhoogd signaal",
    r"sinyal artis",
    r"αυξημενο σημα",
    r"повишен сигнал",
    r"\bthicken",
    r"\bzadebljanje\b",
    r"\bverdikking\b",
    r"\bdistenzij",
    r"\blaksite\b",
    r"\blaxity\b",
    r"\bpartial\b",
    r"\bparcijaln",
    r"\bparcial",
    r"\bpartiel",
    r"\bpartiell",
)

ANAT = {
    "ACL": _rx(
        r"anterior cruciate",
        r"\bacl\b",
        r"cruzado anterior",
        r"\blca\b",
        r"croise anterieur",
        r"voorste kruisband",
        r"\bvkb\b",
        r"vorderes kreuzband",
        r"vorderen kreuzband",
        r"vordere kreuzband",
        r"on capraz",
        r"\bocb\b",
        r"prednji krizni",
        r"prednjeg krizn",
        r"προσθι[οα][^ ]* χιαστ",
        r"προσθιου χιαστου",
        r"χιαστο[^ ]* συνδεσμ",
        r"\bχιαστ\w*",
        r"предна кръстна",
        r"предната кръстна",
        r"cruciate ligaments",
        r"ligamentos cruzados",
        r"ligaments croises",
        r"kruisbanden",
        r"kreuzbander",
        r"capraz baglar",
        r"krizn[a-z]* ligament[a-z]*",
        r"χιαστοι συνδεσμ",
        r"χιαστων συνδεσμ",
        r"кръстните връзки",
        r"кръстни връзки",
    ),
    "MCL": _rx(
        r"medial collateral",
        r"\bmcl\b",
        r"tibial collateral",
        r"colateral medial",
        r"colateral interno",
        r"\blcm\b",
        r"collateral medial",
        r"collateral interne",
        r"mediale collaterale",
        r"binnenband",
        r"\b(mediale|laterale) banden\b",
        r"\bcollaterale banden\b",
        r"innenband",
        r"mediales? kollateral",
        r"\bic yan bag",
        r"medial kollateral",
        r"\biyb\b",
        r"medijalni kolateraln",
        r"medijalnog kolateraln",
        r"εσω πλαγι",
        r"εσωτερικο πλαγι",
        r"\bπλαγι\w* συνδεσμ",
        r"\bπλαγιοι\b",
        r"медиален колатерал",
        r"вътрешна странична",
        r"\bколатерал\w*",
        r"\bcolaterales\b",
        r"\bcollateraux\b",
        r"\bcollateralen\b",
        r"\bkolateralni\b",
        r"collateral ligaments",
        r"ligamentos colaterales",
        r"ligaments collateraux",
        r"collaterale banden",
        r"kollateralbander",
        r"seitenbander",
        r"yan baglar",
        r"kolateraln[a-z]* ligament[a-z]*",
        r"πλαγιοι συνδεσμ",
        r"πλαγιων συνδεσμ",
        r"колатерални връзки",
        r"страничните връзки",
    ),
    "Medial Meniscus": _rx(
        r"medial meniscus",
        r"\bmm\b(?= tear)",
        r"medial menisc",
        r"menisco medial",
        r"menisco interno",
        r"menisque medial",
        r"menisque interne",
        r"mediale meniscus",
        r"binnenmeniscus",
        r"innenmeniskus",
        r"medialen? meniskus",
        r"innenmeniskushinterhorn",
        r"medyal menisk",
        r"\bic menisk",
        r"medijalni meniskus",
        r"medijalnog meniskusa",
        r"medijalnom meniskusu",
        r"εσω μηνισκ",
        r"μηνισκ[^ ]* του εσω",
        r"εσω διαμερισμα[^.]{0,40}μηνισκ",
        r"медиалния менискус",
        r"медиален менискус",
        r"вътрешния менискус",
    ),
    "Lateral Meniscus": _rx(
        r"lateral meniscus",
        r"lateral menisc",
        r"menisco lateral",
        r"menisco externo",
        r"menisque lateral",
        r"menisque externe",
        r"laterale meniscus",
        r"buitenmeniscus",
        r"aussenmeniskus",
        r"lateralen? meniskus",
        r"lateral menisk",
        r"\bdis menisk",
        r"lateralni meniskus",
        r"lateralnog meniskusa",
        r"lateralnom meniskusu",
        r"εξω μηνισκ",
        r"μηνισκ[^ ]* του εξω",
        r"εξω διαμερισμα[^.]{0,40}μηνισκ",
        r"латералния менискус",
        r"латерален менискус",
        r"външния менискус",
    ),
}

OA_EVIDENCE = _rx(
    r"osteoarthrit",
    r"\barthros",
    r"\bgonarthros",
    r"\bosteoarthros",
    r"chondropath",
    r"chondromalac",
    r"condropat",
    r"condromalac",
    r"cartilage loss",
    r"cartilage thinning",
    r"chondral (loss|defect|ulcer|thinning)",
    r"osteophyt",
    r"osteofit",
    r"osteofyt",
    r"osteofito",
    r"osteophyten",
    r"joint space narrowing",
    r"pinzamiento articular",
    r"kikirdak kayb",
    r"kikirdak incelme",
    r"kondropati",
    r"kondral",
    r"kraakbeen(lijden|verlies)",
    r"gonartrose",
    r"artrose",
    r"knorpel(verlust|schaden|defekt)",
    r"arthrose",
    r"gonarthrose",
    r"hrskavic",
    r"hondromalac",
    r"artroz",
    r"osteoartrit",
    r"χονδρ[^ ]*παθ",
    r"αρθριτ",
    r"αρθρωσ",
    r"οστεοφυτ",
    r"αρθρικου χονδρου",
    r"εξαλειψη του αρθρικου χονδρου",
    r"артроз",
    r"хондропат",
    r"остеофит",
    r"хрущял[^.]{0,30}(изтън|увред|дефект)",
    r"ulcera[s]? condral",
    r"cartilago[^.]{0,25}(perdida|adelgaz)",
    r"icrs grade",
    r"outerbridge",
)

COMPARTMENT = {
    "Medial OA": _rx(
        r"medial (femorotibial|tibiofemoral|compartment)",
        r"compartimento femorotibial medial",
        r"femorotibial interno",
        r"mediaal femorotibiaal",
        r"mediale femorotibial",
        r"medial femorotibial",
        r"medialen kompartiment",
        r"innere[sn]? kompartiment",
        r"medyal femorotibial",
        r"ic kompartman",
        r"medyal kompartman",
        r"medijaln[^ ]* (femorotibi|odjelj|kompartm)",
        r"εσω διαμερισμα",
        r"εσω κνημιαι",
        r"εσω μηριαι",
        r"медиалн[^ ]* (компартм|отдел|тибиал|феморотиб)",
        r"medial (femoral|tibial) (condyle|plateau)",
        r"condilo femoral medial",
        r"medialen? (femurkondyl|tibiaplateau)",
        r"mediale femorale condyl",
    ),
    "Lateral OA": _rx(
        r"lateral (femorotibial|tibiofemoral|compartment)",
        r"compartimento femorotibial lateral",
        r"femorotibial externo",
        r"lateraal femorotibiaal",
        r"laterale femorotibial",
        r"lateral femorotibial",
        r"lateralen kompartiment",
        r"aussere[sn]? kompartiment",
        r"lateral femorotibial",
        r"dis kompartman",
        r"lateral kompartman",
        r"lateraln[^ ]* (femorotibi|odjelj|kompartm)",
        r"εξω διαμερισμα",
        r"εξω κνημιαι",
        r"εξω μηριαι",
        r"латералн[^ ]* (компартм|отдел|тибиал|феморотиб)",
        r"lateral (femoral|tibial) (condyle|plateau)",
        r"condilo femoral lateral",
        r"lateralen? (femurkondyl|tibiaplateau)",
        r"laterale femorale condyl",
    ),
    "PF OA": _rx(
        r"patellofemoral",
        r"femoropatellar",
        r"femoropatelar",
        r"patelofemoral",
        r"retropatellar",
        r"retrorotulian",
        r"\btrochlea",
        r"\btroclea",
        r"\btroklea",
        r"\bpatella\b",
        r"\bpatellar\b",
        r"\brotulian",
        r"\brotula\b",
        r"\bpatele\b",
        r"\bpatellae?\b",
        r"patellofemoraal",
        r"femoropatellair",
        r"επιγονατιδ",
        r"μηροεπιγονατιδ",
        r"τροχιλ",
        r"пател",
        r"феморопател",
        r"тролχ",
        r"anterior compartment",
        r"compartimento anterior",
        r"prednj[^ ]* odjeljk",
    ),
}

DIRECT = {
    "Effusion": _rx(
        r"\beffusion",
        r"joint fluid",
        r"intra ?articular fluid",
        r"\bhydrops\b",
        r"derrame articular",
        r"\bderrame\b",
        r"liquido articular",
        r"epanchement",
        r"gewrichtsvocht",
        r"\bvocht\b",
        r"gewrichtseffusie",
        r"gelenkerguss",
        r"\berguss\b",
        r"gelenksergu",
        r"eklem\w* ic\w* sivi",
        r"efuzyon",
        r"eklem sivisi",
        r"sivi (miktari|artisi|birikimi)",
        r"sivi artis",
        r"\bsivi\b[^.]{0,25}artmis",
        r"\bizljev",
        r"\bizliv",
        r"zglobn[^ ]* tekucin",
        r"αρθρικ[^ ]* υγρ",
        r"υγρου ενδαρθρικα",
        r"ενδαρθρικ[^ ]* υγρ",
        r"ποσοτητα υγρου",
        r"ενδαρθρικ",
        r"αρθρικη συλλογη",
        r"υγρο στην αρθρωση",
        r"υγρου στην αρθρωση",
        r"ставен излив",
        r"излив",
        r"ставна течност",
        r"синовиална течност",
    ),
    "Synovitis": _rx(
        r"synovit",
        r"sinovit",
        r"synovial (thickening|proliferation|hypertroph)",
        r"synovitis",
        r"synoviale? (verdikking|proliferatie)",
        r"synovialitis",
        r"synovialis(verdickung|proliferation)",
        r"sinovijalitis",
        r"sinovitis",
        r"zadebljanje sinovij",
        r"υμενιτιδα",
        r"συνοβιτιδα",
        r"υμενικ[^ ]* υπερτροφ",
        r"αρθρικου υμεν",
        r"синовит",
        r"синовиал[^ ]* (задебел|пролифер)",
        r"verdikkingen van (het )?synovium",
        r"pannus",
    ),
    "Baker's": _rx(
        r"baker",
        r"popliteal cyst",
        r"quiste popliteo",
        r"quistes popliteos",
        r"kyste poplite",
        r"popliteale? cyst",
        r"poplitealzyste",
        r"bakerzyste",
        r"popliteal kist",
        r"\bbakerova\b",
        r"poplitealn[^ ]* cist",
        r"κυστη baker",
        r"πολυχωρη συνοβιακη κυστη",
        r"κυστη του baker",
        r"киста на бейкър",
        r"бейкърова киста",
        r"поплитеална киста",
        r"gastrocnemio ?semimembranos",
        r"gastrocnemius semimembranosus burs",
    ),
    "Contusion": _rx(
        r"\bcontusion",
        r"bone bruise",
        r"bone marrow (o?edema|contusion)",
        r"\bkontuz",
        r"medular bone o?edema",
        r"marrow o?edema",
        r"contusion osea",
        r"edema oseo",
        r"edema de medula osea",
        r"oedeme osseux",
        r"contusion osseuse",
        r"botcontusie",
        r"botoedeem",
        r"beenmergoedeem",
        r"botmergoedeem",
        r"knochenmarkodem",
        r"knochenodem",
        r"kontusion",
        r"kemik kontuzyonu",
        r"kemik iligi odemi",
        r"kemik odemi",
        r"kostani edem",
        r"edem kosti",
        r"kontuzij",
        r"οστεομυελικ[^ ]* οιδημα",
        r"οστικο οιδημα",
        r"μυελικο οιδημα",
        r"костномозъчен едем",
        r"костен едем",
        r"контузионен",
    ),
    "Fracture": _rx(
        r"\bfractur",
        r"\bfract\b",
        r"\bfractura",
        r"\bfracturas\b",
        r"\bfractuur",
        r"\bbreuk\b",
        r"\bfraktur",
        r"\bbruch\b",
        r"\bkirik\b",
        r"\bkirigi\b",
        r"\bprijelom",
        r"impresijsk[^ ]* fraktur",
        r"καταγμα",
        r"καταγματ",
        r"фрактур",
        r"счупван",
        r"фисур",
        r"insufficiency fracture",
        r"stress fracture",
        r"avulsion fracture",
        r"subchondral fracture",
        r"subkondral kiri",
    ),
}

DECOY = {
    "Fracture": _rx(r"microfractur", r"\bfracture (risk|prophyla)"),
    "Baker's": _rx(r"meniscal cyst", r"quiste meniscal", r"ganglion"),
}

PAIRED = {"ACL", "MCL", "Medial Meniscus", "Lateral Meniscus"}
OA_TARGETS = {"Medial OA", "Lateral OA", "PF OA"}

STEM_MENISCUS = _rx(r"menisc\w*", r"menisk\w*", r"μηνισκ\w*", r"мениск\w*")
STEM_CRUCIATE = _rx(
    r"cruciate",
    r"cruzado",
    r"croise",
    r"kruisband",
    r"kreuzband",
    r"capraz bag\w*",
    r"krizn\w*",
    r"χιαστ\w*",
    r"кръстн\w*",
    r"\bacl\b",
    r"\bpcl\b",
    r"\blca\b",
    r"\blcp\b",
    r"\bvkb\b",
    r"\bhkb\b",
    r"\bocb\b",
    r"\bacb\b",
)
STEM_COLLATERAL = _rx(
    r"collateral\w*",
    r"colateral\w*",
    r"kollateral\w*",
    r"collaterale\w*",
    r"kolateraln\w*",
    r"yan bag\w*",
    r"πλαγι\w*",
    r"колатерал\w*",
    r"странич\w*",
    r"innenband\w*",
    r"aussenband\w*",
    r"binnenband\w*",
    r"\bmcl\b",
    r"\blcl\b",
    r"\blcm\b",
    r"\biyb\b",
)

SIDE_MEDIAL = _rx(
    r"\bmedial\w*",
    r"\bmedyal\w*",
    r"\bmedijaln\w*",
    r"\bmediaal\w*",
    r"\bmediale\w*",
    r"\bintern[oa]\w*",
    r"\binterne\w*",
    r"\binnen\w*",
    r"\bic\b",
    r"\bunutarnj\w*",
    r"\bεσω\w*",
    r"\bεσωτερικ\w*",
    r"\bмедиал\w*",
    r"\bвътреш\w*",
    r"\btibial collateral\b",
    r"\bbinnen\w*",
    r"\bmediaal\b",
)
SIDE_LATERAL = _rx(
    r"\blateral\w*",
    r"\bextern[oa]\w*",
    r"\bexterne\w*",
    r"\bdis\b",
    r"\blateraln\w*",
    r"\baussen\w*",
    r"\bbuiten\w*",
    r"\bεξω\w*",
    r"\bεξωτερικ\w*",
    r"\bлатерал\w*",
    r"\bвъншн\w*",
    r"\bfibular collateral\b",
    r"\bvanjsk\w*",
)
SIDE_ANTERIOR = _rx(
    r"\banterior\w*",
    r"\bant\b",
    r"\bon\b",
    r"\bprednj\w*",
    r"\bvorder\w*",
    r"\bvoorste\b",
    r"\bπροσθι\w*",
    r"\bпредн\w*",
    r"\banteriyor\w*",
    r"\bavant\b",
    r"\bant[eé]rieur\w*",
)
SIDE_POSTERIOR = _rx(
    r"\bposterior\w*",
    r"\bpost[eé]rieur\w*",
    r"\bposteriore\w*",
    r"\bhinter\w*",
    r"\bachterste\b",
    r"\barka\b",
    r"\bstraznj\w*",
    r"\bzadnj\w*",
    r"\bοπισθι\w*",
    r"\bзадн\w*",
    r"\bpostero\w*",
)

STEM_FRACTURE = _rx(
    r"fractur\w*",
    r"fraktur\w*",
    r"fractuur\w*",
    r"\bfract\b",
    r"kiri[kgğ]\w*",
    r"prijelom\w*",
    r"lom kosti",
    r"\bbreuk\w*",
    r"\bbruch\w*",
    r"καταγμα\w*",
    r"καταγματ\w*",
    r"фрактур\w*",
    r"счупван\w*",
    r"fisur\w* (osea|oseas|kost)",
    r"fissur\w* kost",
)

STEM_OA_COMPARTMENT = _rx(
    r"compartment\w*",
    r"compartimento\w*",
    r"compartiment\w*",
    r"kompartman\w*",
    r"kompartiment\w*",
    r"odjelj\w*",
    r"διαμερισμα\w*",
    r"компартм\w*",
    r"\bотдел\w*",
    r"femorotibial\w*",
    r"femorotibiaal\w*",
    r"tibiofemoral\w*",
    r"femoro tibial\w*",
    r"κνημιαι\w*",
    r"μηριαι\w*",
    r"femoral condyl\w*",
    r"tibial plateau\w*",
    r"condilo femoral",
    r"platillo tibial",
    r"tibiaplateau\w*",
    r"femurkondyl\w*",
    r"femoralne? kondil\w*",
    r"tibijaln\w* plato",
    r"femoral kondil\w*",
    r"tibia plato",
    r"tibyal plato",
)


def _distance(
    clause: str, stem_rx: re.Pattern[str], qual_rx: re.Pattern[str], window: int = 55
) -> int | None:
    best = None
    for m in stem_rx.finditer(clause):
        lo = max(0, m.start() - window)
        hi = min(len(clause), m.end() + window)
        for q in qual_rx.finditer(clause[lo:hi]):
            qs, qe = lo + q.start(), lo + q.end()
            d = 0 if qs < m.end() and qe > m.start() else min(abs(m.start() - qe), abs(qs - m.end()))
            best = d if best is None else min(best, d)
    return best


def _near(clause: str, stem_rx: re.Pattern[str], qual_rx: re.Pattern[str], window: int = 55) -> bool:
    return _distance(clause, stem_rx, qual_rx, window) is not None


STEM_RULES = {
    "ACL": (STEM_CRUCIATE, SIDE_ANTERIOR),
    "MCL": (STEM_COLLATERAL, SIDE_MEDIAL),
    "Medial Meniscus": (STEM_MENISCUS, SIDE_MEDIAL),
    "Lateral Meniscus": (STEM_MENISCUS, SIDE_LATERAL),
    "Medial OA": (STEM_OA_COMPARTMENT, SIDE_MEDIAL),
    "Lateral OA": (STEM_OA_COMPARTMENT, SIDE_LATERAL),
}

SEV_LOW = _rx(
    r"\bsmall\b",
    r"\bminimal\b",
    r"\btrace\b",
    r"\bmild\b",
    r"\bslight\b",
    r"\btiny\b",
    r"\bscant\b",
    r"\bmimimal\b",
    r"\bdiscrete\b",
    r"\bfocal\b",
    r"\bleve\b",
    r"\bminim",
    r"\bpeque",
    r"\bligero\b",
    r"\bescaso\b",
    r"\bdiscreto\b",
    r"\bhafif\b",
    r"\bminimal\b",
    r"\baz miktarda\b",
    r"\bsilik\b",
    r"\bmanja\b",
    r"\bmanji\b",
    r"\bblago\b",
    r"\bdiskretn",
    r"\bmalo\b",
    r"\bgering",
    r"\bdiskret",
    r"\bkleine?r?\b",
    r"\bwenig\b",
    r"\bzarte?\b",
    r"\bbeperkte?\b",
    r"\bgeringe\b",
    r"\bweinig\b",
    r"\blichte?\b",
    r"\bηπι",
    r"\bμικρ",
    r"\bελαχιστ",
    r"\bминимал",
    r"\bлек",
    r"\bмалк",
    r"\bнеголям",
)

SEV_HIGH = _rx(
    r"\blarge\b",
    r"\bmarked\b",
    r"\bmassive\b",
    r"\bsevere\b",
    r"\bextensive\b",
    r"\bmoderate\b",
    r"\bgross\b",
    r"\bsignificant\b",
    r"\babundant\b",
    r"\btense\b",
    r"\bmoderad",
    r"\bimportante\b",
    r"\bsevera?\b",
    r"\bmarcad",
    r"\bcuantios",
    r"\bbelirgin\b",
    r"\byaygin\b",
    r"\bileri\b",
    r"\bciddi\b",
    r"\bbol\b",
    r"\bopsezan\b",
    r"\bveliki\b",
    r"\bizrazit",
    r"\bznacajn",
    r"\bumjeren",
    r"\bausgepragt",
    r"\bdeutlich",
    r"\bmassiv",
    r"\bmassig",
    r"\bgross",
    r"\buitgebreid",
    r"\bgevorderd",
    r"\bveel\b",
    r"\bmatige?\b",
    r"\bμετρι",
    r"\bμεγαλ",
    r"\bεκτεταμεν",
    r"\bευμεγεθ",
    r"\bσοβαρ",
    r"\bголям",
    r"\bизразен",
    r"\bзначим",
    r"\bумерен",
    r"\bобилен",
)

GLOBAL_OA = _rx(
    r"tri ?compartment",
    r"all three compartment",
    r"global(ised)? (oa|osteoarthrit)",
    r"\bgonarthros",
    r"\bgonartros",
    r"\bgonarthrose",
    r"\bgonartrose",
    r"osteoarthritis of the knee",
    r"artrosis (de |)(la )?rodilla",
    r"knee osteoarthrit",
    r"\bdiz osteoartrit",
    r"\bgonartroz",
    r"artroza koljena",
    r"οστεοαρθριτιδα",
    r"αρθριτιδα του γονατος",
    r"артроза на колянната",
    r"гонартроз",
    r"degenerative joint disease",
    r"\bdjd\b",
)

DEGENERATIVE_MARROW = _rx(
    r"subchondral",
    r"subcondral",
    r"subkondral",
    r"supkondraln",
    r"subchondraln",
    r"υποχονδρι",
    r"субхондрал",
    r"subchondrale?",
    r"\bcyst",
    r"\bquist",
    r"\bzyste\b",
    r"\bcistic",
    r"reactive",
    r"reactivo",
)

TRAUMA = _rx(
    r"\bbruise\b",
    r"\bcontusion",
    r"\bkontuz",
    r"\bcontusion osea\b",
    r"\btrauma",
    r"\bimpaction\b",
    r"\bpivot shift\b",
    r"\bkissing\b",
    r"\bacute\b",
    r"\bagudo\b",
    r"\bakut",
    r"\bpivot kaymasi\b",
    r"\bcontusion osseuse\b",
    r"\bbone bruise\b",
    r"\bbotcontusie\b",
    r"\bконтузион",
    r"\bμωλωπ",
    r"\bkontuzij",
)


def _polarity(clause: str, anchor_end: int) -> str:
    if UNCERTAIN.search(clause):
        return "uncertain"
    if NEGATION.search(clause):
        return "negative"
    if NORMALITY.search(clause):
        if TEAR.search(clause) or re.search(r"\bgrade [34]\b", clause):
            return "positive"
        return "negative"
    return "positive"


class _Matcher:
    def __init__(
        self,
        phrase_rx: re.Pattern[str],
        stem: re.Pattern[str] | None = None,
        side: re.Pattern[str] | None = None,
        window: int = 55,
        contrary: re.Pattern[str] | None = None,
    ):
        self.phrase_rx = phrase_rx
        self.stem = stem
        self.side = side
        self.window = window
        self.contrary = contrary

    def search(self, clause: str) -> re.Match[str] | None:
        m = self.phrase_rx.search(clause)
        if m is not None and not self._wrong_side(clause):
            return m
        if self.stem is not None and self.side is not None and _near(clause, self.stem, self.side, self.window):
            return self.stem.search(clause)
        return None

    def _wrong_side(self, clause: str) -> bool:
        if self.contrary is None or self.stem is None or self.side is None:
            return False
        other = _distance(clause, self.stem, self.contrary, self.window)
        if other is None:
            return False
        own = _distance(clause, self.stem, self.side, self.window)
        return own is None or other < own


CONTRARY = {"ACL": SIDE_POSTERIOR, "MCL": SIDE_LATERAL}

ANAT_MATCH = {
    tgt: _Matcher(ANAT[tgt], *STEM_RULES[tgt], contrary=CONTRARY.get(tgt)) for tgt in PAIRED
}
COMPARTMENT_MATCH = {
    "Medial OA": _Matcher(COMPARTMENT["Medial OA"], *STEM_RULES["Medial OA"]),
    "Lateral OA": _Matcher(COMPARTMENT["Lateral OA"], *STEM_RULES["Lateral OA"]),
    "PF OA": _Matcher(COMPARTMENT["PF OA"]),
}
DIRECT_MATCH = {
    tgt: _Matcher(_rx(rx.pattern, STEM_FRACTURE.pattern) if tgt == "Fracture" else rx)
    for tgt, rx in DIRECT.items()
}


def _severity(clause: str) -> float:
    high = SEV_HIGH.search(clause) is not None
    low = SEV_LOW.search(clause) is not None
    if high and not low:
        return 1.0
    if low and not high:
        return 0.45
    return 0.75


POSTOP_RESECTION = _rx(r"meniscectom\w*", r"resection\b", r"status post\b", r"\bs/p\b", r"prior repair")


def _score_clauses(
    cls: list[str],
    anat_rx: _Matcher,
    path_rx: re.Pattern[str] | None = None,
    decoy_rx: re.Pattern[str] | None = None,
    context_penalty: re.Pattern[str] | None = None,
    context_bonus: re.Pattern[str] | None = None,
) -> tuple[float, float, int, int]:
    n_pos = n_neg = n_unc = 0
    best = 0.0
    for c in cls:
        m = anat_rx.search(c)
        if not m:
            continue
        if decoy_rx is not None and decoy_rx.search(c):
            continue
        if POSTOP_RESECTION.search(c) and not TEAR.search(c):
            n_neg += 1
            continue
        if path_rx is not None and not path_rx.search(c):
            if NORMALITY.search(c) and not NEGATION.search(c):
                n_neg += 1
            continue
        pol = _polarity(c, m.end())
        if pol == "positive":
            n_pos += 1
            w = _severity(c)
            if context_penalty is not None and context_penalty.search(c):
                w *= 0.45
            if context_bonus is not None and context_bonus.search(c):
                w = min(1.0, w * 1.35)
            best = max(best, w)
        elif pol == "negative":
            n_neg += 1
        else:
            n_unc += 1
            best = max(best, 0.30)

    if n_pos or n_unc:
        score = min(0.95, 0.50 + 0.42 * best + 0.03 * min(n_pos, 3))
        conf = min(1.0, 0.55 + 0.15 * n_pos)
    elif n_neg:
        score = max(0.04, 0.20 - 0.04 * n_neg)
        conf = min(0.9, 0.45 + 0.12 * n_neg)
    else:
        score, conf = 0.28, 0.05
    return score, conf, n_pos, n_neg


def extract_from_report(report: str) -> dict[str, float]:
    """Extract twelve continuous confidence scores in [0, 1] from one report text."""
    cls = clauses(report)
    out: dict[str, float] = {}
    path_paired = _rx(TEAR.pattern, DEGEN.pattern, INJURY.pattern)

    for tgt in TARGETS:
        if tgt in PAIRED:
            s, c, npos, nneg = _score_clauses(cls, ANAT_MATCH[tgt], path_paired)
        elif tgt in OA_TARGETS:
            s, c, npos, nneg = _score_clauses(cls, COMPARTMENT_MATCH[tgt], OA_EVIDENCE)
        elif tgt == "Contusion":
            s, c, npos, nneg = _score_clauses(
                cls,
                DIRECT_MATCH[tgt],
                None,
                DECOY.get(tgt),
                context_penalty=DEGENERATIVE_MARROW,
                context_bonus=TRAUMA,
            )
        else:
            s, c, npos, nneg = _score_clauses(cls, DIRECT_MATCH[tgt], None, DECOY.get(tgt))
        out[tgt] = s
        out[tgt + "__conf"] = c
        out[tgt + "__npos"] = float(npos)
        out[tgt + "__nneg"] = float(nneg)

    g_hits = [c for c in cls if GLOBAL_OA.search(c) and _polarity(c, 0) == "positive"]
    if g_hits:
        gscore = 0.50 + 0.42 * max(_severity(c) for c in g_hits)
        for tgt in OA_TARGETS:
            if out[tgt + "__npos"] == 0 and out[tgt + "__nneg"] == 0:
                out[tgt] = max(out[tgt], gscore * 0.92)
                out[tgt + "__conf"] = max(out[tgt + "__conf"], 0.4)

    if out["Synovitis__npos"] == 0 and out["Synovitis__nneg"] == 0:
        out["Synovitis"] = max(out["Synovitis"], 0.28 + 0.45 * (out["Effusion"] - 0.28))

    return out


def populate_labels_from_reports(
    df: pd.DataFrame, target_cols: Sequence[str] = TARGETS
) -> pd.DataFrame:
    """Populate missing targets from reports while strictly preserving human gold labels.

    Args:
        df: DataFrame containing 'StudyInstanceUID', 'Report', and optional target columns.
        target_cols: List of 12 targets.

    Returns:
        DataFrame with complete targets and confidence weights.
    """
    df = df.copy()
    extracted_records = []
    for report in df["Report"]:
        extracted_records.append(extract_from_report(str(report)))

    ext_df = pd.DataFrame(extracted_records, index=df.index)

    for col in target_cols:
        # Where human gold label exists and is not null, keep it; otherwise fill from extracted score
        if col in df.columns:
            df[col] = df[col].combine_first(ext_df[col])
        else:
            df[col] = ext_df[col]

        # Save confidence weight
        df[f"{col}__conf"] = ext_df[f"{col}__conf"]

    return df
