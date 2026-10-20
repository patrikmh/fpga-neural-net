# BROAD: sammanhängande träning

Två förregistrerade körningar med seeds 7 och 17 fortsatte från den valda
H96-checkpointen. Varje körning gick tio gånger genom alla fyra
träningsinspelningarna, med 5 390 optimeringssteg. Minnet följde hela
inspelningen och kopplades bort från gradientgrafen efter varje block om
256 prov. Minnet återställdes mellan inspelningar.

Ingen körning förbättrade startmodellens valideringsresultat. Epok noll
var därför bäst i båda fallen. Den ursprungliga checkpointen är fortsatt
vald: **26,433936° fullständigt orienterings-RMSE**, 11,827967° lutning och
23,794987° heading, som medelvärde över valideringsinspelningarna 02/30.
Försöket visar ingen förbättring med detta träningsupplägg; det avgör inte
om andra upplägg kan fungera bättre.

Vald checkpoint är `artifacts/broad-width-study-v1/checkpoint.pt`, SHA256
`a936cd787bf0db0969621a0da548bb044b384e0f1679bdc83bdf611d5f8e1ba9`.
De nya körningarnas epok-noll-checkpointer har annan metadata och andra
filhashar, men ersätter inte denna checkpoint.

## Underlag och reproduktion

[Förregistrerat upplägg](../experiments/broad-stateful-study-v1.json) och
[sparade rapporter](../artifacts/broad-stateful-study-v1/) binder data,
startcheckpoint, källfiler, miljö och urval. `run-study.py` i artefaktmappen
är den använda köraren och kräver samma förberedda träningsmanifest.
Källorna för den ursprungliga flyttalsmodellen ändrades inte.

```sh
PYTHONPATH=. .venv/bin/python artifacts/broad-stateful-study-v1/run-study.py
```

Köraren kräver nya utmappar. Den utför båda körningarna enligt planen,
kontrollerar uppdateringsantal och räknar självständigt om måtten från
sparade prediktioner.

CLI:n verifierades även separat på syntetiska data: två epoker, tio
optimeringssteg, ändliga normerade quaternioner och identiska prediktioner
efter omladdning. Den tidigare fulla sviten med 77 tester passerade efter
implementationen; fyra ytterligare gränsfall passerade tillsammans med
det ursprungliga integrationstestet.

Nästa steg är en separat heltalsreferens, därefter RTL, faktisk FPGA-passning
och fysisk verifiering. Gränsen för kvantiseringsförlust är låst före
utvärdering i [heltalsplanen](../experiments/broad-numeric-study-v1.json).
Nya sluttestinspelningar 04/08/12/31 är fortfarande oöppnade.
