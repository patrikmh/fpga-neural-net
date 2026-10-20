# BROAD: första CfC-försöket

Det separata CPU-försöket är genomfört med nio sensorkanaler, åtta CfC-tillstånd
och fyra normaliserade quaternionutgångar: 612 parametrar. Valideringsfelet
blev 34,71° i fullständigt orienterings-RMSE. Den förutbestämda budgeten var
50 epoker, 16 uppdateringar per epok och seed 7; epok 50 valdes på validering.

Checkpointen låstes innan sluttestets fyra inspelningar hämtades. Kod,
rotationsmått och datahashar låstes före konvertering och utvärdering.
Inga inställningar ändrades efter sluttestet.

| Sluttest | Full orientering CfC | Lutning CfC | Lutning gyro + accel |
|---|---:|---:|---:|
| 03 långsam rotation C | 38,03° | 18,80° | 0,89° |
| 07 snabb rotation B | 58,65° | 27,31° | 2,95° |
| 11 långsam translation B | 7,34° | 3,54° | 4,42° |
| 29 stationär magnet B | 63,89° | 27,89° | 10,43° |
| **Medel av inspelningarnas RMSE** | **41,98°** | **19,39°** | **4,67°** |

CfC:s genomsnittliga heading-RMSE var 38,01°. Baslinjen skattar bara lutning
och har därför inget fullständigt orienterings- eller heading-resultat.
136 580 rörelseprov med giltigt optiskt facit poängsattes. 513 rörelseprov
saknade giltigt facit och undantogs utan att tidsstegen togs bort.
Alla 217 588 testtidssteg användes för kontinuerlig tillståndsuppdatering.

Den här åtta-neuronsmodellen klarar inte kvalitetskontrollen för nästa FPGA-steg:
lutningsfelet är större än den enkla kausala baslinjens. Detta är ett litet
tekniskt försök med fyra testinspelningar, inte hela BROAD-benchmarken.
Patrik har därefter beställt en större modell som utnyttjar kortets kapacitet.
De fyra redan granskade testinspelningarna måste då benämnas återanvända
testdata, inte ett nytt orört sluttest.

## Sparade underlag

[Artefaktkatalogen](../artifacts/broad-cfc8-v1/) innehåller checkpoint,
träningsrapport, urvalslåsning, fryst testspecifikation, konverteringsmanifest,
slutrapport och sammanfattning. Rådata och fullständiga prediktionsfiler ligger
lokalt i `build/`. Ursprunglig träningskonverterare är sparad med verifierad
SHA-256; en ny körning återskapade träningsmanifestet byte för byte.

Datakälla är [BROAD](https://github.com/dlaidig/broad), version
`7e2f8189bb4a3ac91eae32c69b6bc87f08a6e2ad`, CC BY 4.0:
D. Laidig, M. Caruso, A. Cereatti och T. Seel (2021),
[BROAD—A Benchmark for Robust Inertial Orientation Estimation](https://doi.org/10.3390/data6070072).
Indata är acc/gyr/mag, referens är wxyz body-to-ENU, takt 2000/7 Hz.
Baslinjen är vår egen Rodrigues-integration med accelerometerkorrigering,
tidskonstant 0,5 s och initiering från accelerometern, utan referensbaserad
inriktning. Fullständiga feldefinitioner finns i den frysta specifikationen.

## Reproducera

Den publicerade H8-körningen hör till Git-version `d8d5095`. Checka ut den
versionen för reproduktion: senare breddstudier ändrar källhashar, vilket den
frysta utvärderingen avsiktligt avvisar. Behåll projektets låsta `.venv`.
HDF5-konvertering använder separat miljö:

```sh
uv venv build/broad-analysis-env --python .venv/bin/python
uv pip install --python build/broad-analysis-env/bin/python h5py==3.16.0 numpy==2.4.6
```

Hämta filerna i `trials` från specifikationen till en ny datakatalog. Exakt URL
är `https://raw.githubusercontent.com/dlaidig/broad/<source_revision>/data_hdf5/<file>`.
Konverteraren kontrollerar varje fils SHA-256 före läsning.

För den publicerade checkpointens sluttest, använd exempelvis `build/broad-test-data`:

```sh
build/broad-analysis-env/bin/python -m experiments.broad_prepare --data-dir build/broad-test-data --spec artifacts/broad-cfc8-v1/test_spec.json --output-dir build/broad-test-prepared --split test
.venv/bin/python -m model.broad_evaluate --checkpoint artifacts/broad-cfc8-v1/checkpoint.pt --prepared-dir build/broad-test-prepared --spec artifacts/broad-cfc8-v1/test_spec.json --output-dir build/broad-test-result
```

Utdatamappar måste vara nya eller tomma. Utvärderingen avvisar avvikande
checkpoint, kod, Python-/paketversioner och dataidentiteter före inferens.
För en ny träning används `experiments/broad-train-spec-v1.json` och ursprunglig
konverterare för att återskapa det publicerade träningsmanifestet exakt:

```sh
build/broad-analysis-env/bin/python artifacts/broad-cfc8-v1/training_converter.py --data-dir build/broad-train-data --spec experiments/broad-train-spec-v1.json --output-dir build/broad-train-prepared
.venv/bin/python -m model.broad_train --prepared-dir build/broad-train-prepared --output-dir build/broad-training
```

Verifiering: tre TDD-cykler, 73 godkända tester på 156,983 s och compileall.
Oberoende CLI-smoke körde riktiga syntetiska HDF5-filer genom konvertering,
träning, låsning och slututvärdering; samtliga steg passerade. Det kontrollerade
även normerade quaternioner, maskerade referensluckor och skydd mot återanvänd
utdata. Det tidigare 600-sekunders USB-stabilitetskravet är fortfarande öppet.
