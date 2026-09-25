import json
from pathlib import Path

path = Path("MNIST_EDA_Uitgewerkt.ipynb")
nb = json.loads(path.read_text(encoding="utf-8-sig"))

summary = None
for cell in nb["cells"]:
    for output in cell.get("outputs", []):
        for line in output.get("text", []):
            if line.startswith("RESULT_JSON:"):
                summary = json.loads(line[len("RESULT_JSON:"):])

if summary is None:
    raise RuntimeError("Trainingsresultaten ontbreken.")

s = summary
confusions = sorted(s["confusions"], key=lambda x: x[2], reverse=True)
a, b, count = confusions[0]

answers = {
    9: "Er zijn 784 pixelkolommen en één extra kolom voor het label.",
    18: "De percentages in train en test zijn vrijwel gelijk. "
        "Dat is het doel van stratified sampling.",
    25: "De klasseverdeling is redelijk evenwichtig. "
        "Direct oversamplen is daarom niet nodig.",
    28: "De cijfers verschillen in handschrift, dikte en positie. "
        "Die variatie kan de herkenning moeilijker maken.",
    39: f"""| Mogelijke stap | Nodig? | Reden |
|---|---|---|
| Missing values imputeren | {'Ja' if s['missing'] else 'Nee'} | {s['missing']} ontbrekende waarden |
| Normaliseren | Ja | Pixelwaarden omzetten naar 0–1 |
| Oversampling | Nee | Klassen zijn redelijk evenwichtig |
| One-hot encoding pixels | Nee | Pixels zijn numeriek |
| Data augmentation | Optioneel | Kan generalisatie verbeteren |""",
    42: "Delen door 255 gebruikt een vooraf bekende vaste schaal. "
        "Een StandardScaler moet parameters uitsluitend op de trainingsdata leren.",
    43: f"""De dataset heeft {s['shape'][0]} afbeeldingen en
{s['shape'][1]} kolommen, inclusief het label.
Er zijn {s['missing']} ontbrekende waarden en
{s['duplicates']} exacte duplicaten in de trainingsdata.
We normaliseren de pixels naar 0–1.
Een belangrijk risico buiten MNIST is dat echte handgeschreven
cijfers anders kunnen zijn dan de trainingsvoorbeelden.""",
    53: "Er zijn tien output-units omdat MNIST tien klassen bevat: 0 t/m 9.",
    60: f"""De uiteindelijke trainingsaccuracy is
{s['train_accuracy']:.2%} en de validation accuracy
{s['val_accuracy']:.2%}.
Bekijk de learning curves om te beoordelen of er overfitting optreedt.""",
    69: f"""Het model verwart onder andere {a} met {b}.
Dit gebeurde {count} keer in de testset.""",
    72: f"""De EDA laat zien dat MNIST {s['shape'][0]} afbeeldingen
bevat met tien cijferklassen. De pixelwaarden zijn genormaliseerd
naar 0–1. We gebruikten afzonderlijke train-, validation- en
testsets om datalekken en te optimistische evaluatie te voorkomen.
Het neurale netwerk behaalde {s['test_accuracy']:.2%} accuracy
op de testset, tegenover {s['majority_baseline']:.2%} voor de
majority baseline. Een voorbeeld van een verwarring is
{a} voorspellen bij een echte {b}, of omgekeerd.
MNIST is relatief schoon en representatief voor echte
handschrifttoepassingen is daarmee niet gegarandeerd."""
}

for index, answer in answers.items():
    nb["cells"][index]["source"] = (
        "".join(nb["cells"][index]["source"]).rstrip()
        + "\n\n**Uitwerking:**\n\n" + answer + "\n"
    ).splitlines(True)

path.write_text(
    json.dumps(nb, ensure_ascii=False, indent=1),
    encoding="utf-8"
)

print(f"Test accuracy: {s['test_accuracy']:.2%}")
print(f"Majority baseline: {s['majority_baseline']:.2%}")
print(f"Meest voorkomende verwarring: {a} / {b} ({count} keer)")
print("Resultaatafhankelijke antwoorden toegevoegd.")
