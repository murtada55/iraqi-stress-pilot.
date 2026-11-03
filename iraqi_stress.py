"""
Stress Assignment in Iraqi Arabic: A Pilot Computational Study
Murtadha Al Hasani

This script runs the baseline analysis and tests the four hypotheses (H1-H4) 
for the 52-word dataset. 

Usage: python iraqi_stress.py
"""

from collections import defaultdict

# CV conversion variables
VOWELS = set("aiueo")
DIGRAPHS = ["kh", "sh", "gh", "ch", "th"]

def syllable_to_cv(syll):
    # Convert a syllable to its CV shape
    syll = syll.lower()
    cv_string = ""
    i = 0
    while i < len(syll):
        if i < len(syll) - 1 and syll[i:i + 2] in DIGRAPHS:
            cv_string += "c"
            i += 2
        elif syll[i] in VOWELS:
            cv_string += "v"
            i += 1
        else:
            cv_string += "c"
            i += 1
    return cv_string

def cv_pattern(ipa):
    # Get the full CV shape of a word
    return ".".join(syllable_to_cv(s) for s in ipa.split("."))

def predict_stress(cv_sequence):
    # The 6 stress rules from Section 4.2
    syllables = cv_sequence.split(".")
    n = len(syllables)

    if syllables[-1] in ["cvvc", "cvcc"]:
        return n - 1
    if syllables[-1] == "cvv":
        return n - 1
    if n >= 2 and syllables[-2] in ["cvc", "cvv"]:
        return n - 2
    if n == 2 and syllables[0] == "cv" and syllables[1] == "cv":
        return 0
    if n >= 3 and syllables[-3] in ["cvc", "cvv"]:
        return n - 3
    
    return 0

# Dataset: (Arabic word, transcription, stress index)
DATA = [
    ("ماي", "maay", 0),
    ("سوگ", "soog", 0),
    ("چبير", "chi.biir", 0),
    ("اصغير", "'iS.ghiir", 0),
    ("مروحا", "mroo.ha", 1),
    ("رايحة", "raay.Ha", 1),
    ("شغلة", "shugh.la", 0),
    ("ريوگ", "ri.yoog", 1),
    ("سماچ", "si.maach", 1),
    ("سبانا", "sbaa.na", 1),
    ("ايدرس", "'ii.da.ris", 1),
    ("شتاكل", "she.ta.kil", 0),
    ("عرس", "3i.ris", 1),
    ("تبليط", "tib.liit", 1),
    ("تنور", "tan.nuur", 1),
    ("كاشي", "kaa.shi", 1),
    ("مرگه", "mar.ga", 1),
    ("شراي", "shraay", 0),
    ("گلاص", "gul.laas", 0),
    ("حايط", "Haa.yiT", 0),
    ("مخده", "makh.de", 1),
    ("زوليه", "zoo.li.ya", 1),
    ("سطح", "sa.TaH", 1),
    ("قبط", "qa.bat", 0),
    ("بيتونه", "bee.too.na", 1),
    ("وگفه", "wug.fa", 1),
    ("بزونه", "bzoo.na", 1),
    ("بگال", "bi.gaal", 1),
    ("سيبيندي", "see.been.di", 0),
    ("هياتهه", "hi.yaa.ti.ha", 2),
    ("هاذيا", "haa.thi.ya", 1),
    ("بجامه", "bij.jaa.ma", 0),
    ("ثرية", "tha.ri.ya", 1),
    ("برده", "bar.de", 1),
    ("سمسمية", "sim.si.mi.ya", 2),
    ("بنچرچي", "bin.char.chi", 0),
    ("مجنبز", "mij.ni.biz", 1),
    ("گهاوي", "ga.haa.wi", 0),
    ("ميصير", "mii.Seer", 0),
    ("جناي", "ji.naay", 1),
    ("جنطة", "jin.Ta", 0),
    ("طرگاعة", "tar.gaa.3a", 0),
    ("كلاوات", "ka.la.wat", 0),
    ("مسودن", "mi.soo.din", 2),
    ("بوشية", "bu.shi.ya", 1),
    ("يفوخ", "yi.foox", 1),
    ("حكاكة", "Ha.kaa.ka", 1),
    ("إشسويت", "'ish.sa.weet", 0),
    ("سبتتنگ", "sib.ti.tang", 2),
    ("سختـچي", "sakh.chi", 0),
    ("كلاوچـي", "ka.law.chi", 1),
    ("كلبچة", "ka.lab.cha", 2),
]

# Words used for the Borrowing hypothesis (H3)
BORROWED = [
    "gul.laas", "see.been.di", "bij.jaa.ma", "bin.char.chi",
    "sib.ti.tang", "sakh.chi", "ka.law.chi", "ka.lab.cha",
]

# Words with epenthesis removed for H1
UNDERLYING = {
    "shugh.la": "shgh.la",
    "makh.de": "mkh.de",
    "wug.fa": "wg.fa",
}

def load_words():
    words = []
    for i, (arabic, ipa, stress) in enumerate(DATA, start=1):
        words.append({
            "id": i, 
            "arabic": arabic, 
            "ipa": ipa,
            "syllables": ipa.split("."), 
            "stress": stress
        })
    return words

def self_check(words):
    problems = []
    for w in words:
        for syll in w["syllables"]:
            if not any(ch in VOWELS for ch in syll.lower()):
                problems.append(f"#{w['id']} {w['ipa']}: syllable '{syll}' has no vowel")
        if not 0 <= w["stress"] < len(w["syllables"]):
            problems.append(f"#{w['id']} {w['ipa']}: stress index out of bounds")
    return problems

def test_groups(groups):
    testable = {k: v for k, v in groups.items() if len(v) >= 2}
    if not testable:
        return "not testable"
    
    is_supported = False
    for key, stresses in sorted(testable.items()):
        print(f"  {key}: n={len(stresses)}, stresses={stresses}")
        if len(set(stresses)) == 1:
            is_supported = True
            
    return "supported" if is_supported else "refuted"

def count_mixed(patterns):
    multi = {k: v for k, v in patterns.items() if len(v) >= 2}
    mixed = {k: v for k, v in multi.items() if len(set(v)) > 1}
    return len(mixed), len(multi)

def run_baseline(words):
    correct, errors, left_shifts = 0, 0, 0
    for w in words:
        predicted = predict_stress(cv_pattern(w["ipa"]))
        actual = w["stress"]
        if predicted == actual:
            correct += 1
        else:
            errors += 1
            if predicted < actual:
                left_shifts += 1

    print(f"5.1 Baseline: {correct/len(words)*100:.1f}% ({correct}/{len(words)})")
    print(f"Errors: {errors}, Left shifts: {left_shifts} ({left_shifts/errors*100:.0f}% of errors)")

def run_h1(words):
    surface = defaultdict(list)
    for w in words:
        surface[cv_pattern(w["ipa"])].append(w["stress"])
    mixed, total = count_mixed(surface)
    print(f"\n5.2 H1: Surface {mixed}/{total} mixed")
    
    underlying = defaultdict(list)
    for w in words:
        underlying[cv_pattern(UNDERLYING.get(w["ipa"], w["ipa"]))].append(w["stress"])
    mixed_u, total_u = count_mixed(underlying)
    print(f"Underlying: {mixed_u}/{total_u} mixed")

def first_vowel(ipa):
    for ch in ipa.lower():
        if ch in VOWELS:
            return ch

def run_h2(words):
    groups = defaultdict(list)
    for w in words:
        groups[first_vowel(w["ipa"])].append(w["stress"])
    print("\n5.3 H2")
    print("  Verdict:", test_groups(groups))

def run_h3(words):
    by_ipa = {w["ipa"]: w["stress"] for w in words}
    stresses = [by_ipa[b] for b in BORROWED]
    print("\n5.4 H3")
    print("  Verdict:", test_groups({"borrowed": stresses}))

def run_h4(words):
    groups = defaultdict(list)
    for w in words:
        if len(w["syllables"]) == 3:
            groups[syllable_to_cv(w["syllables"][1]).upper()].append(w["stress"])
    print("\n5.5 H4")
    print("  Verdict:", test_groups(groups))

if __name__ == "__main__":
    words = load_words()
    issues = self_check(words)
    
    if issues:
        print("Self-check failed:")
        for issue in issues:
            print(" -", issue)
        raise SystemExit(1)
        
    print(f"Self-check passed ({len(words)} words).")
    run_baseline(words)
    run_h1(words)
    run_h2(words)
    run_h3(words)
    run_h4(words)