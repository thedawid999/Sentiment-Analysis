# ⭐ Sentimentanalyse von Produktrezensionen
***
## 👤 Projektinformationen

| **Autor** | thedawid999 |
| :--- | :--- |
| **Studiengang** | Angewandte Künstliche Intelligenz |
| **Projekt/Modul** | Natural Language Processing |

#### Verwendete Libraries
* `matplotlib`
* `pandas`
* `numpy`
* `scikit-learn`
* `nltk`
* `joblib`
* `re`
* `string`

***

## 🌟 Projektziel

Ziel dieses Projekts ist die Entwicklung und Evaluation von **klassischen Machine-Learning-Verfahren** zur **multiklassigen Sentimentanalyse** von Produktrezensionen.  
Auf Basis reiner Textdaten sollten **Sternebewertungen von 1 bis 5** vorhergesagt werden.

Der Fokus lag dabei auf:
 * Vergleich zweier etablierter Klassifikationsverfahren
 * Analyse der Grenzen klassischer Textklassifikation
 * Interpretation der Ergebnisse

Das Projekt verdeutlicht, warum selbst gut performende Modelle bei subjektiv annotierten Textdaten an natürliche Grenzen stoßen.

***

## 📊 Datensatz

**Quelle**: Amazon Reviews 2023 Dataset

**Zielvariable**: Sternebewertungen (1–5)

**Link:** [Amazon Reviews](https://amazon-reviews-2023.github.io/index.html)

**Gewählte Kategorien**:
 * AllBeauty (701.528 Rezensionen)
 * HandmadeProducts (664.162 Rezensionen)
 * HealthAndPersonalCare (494.121 Rezensionen)

***

## 🔍 Explorative Datenanalyse (EDA)

Die explorative Analyse umfasste:
* Untersuchung der Datentypen und Attributstruktur
* Analyse der Klassenverteilung der Sternebewertungen (1–5)
* Überprüfung auf fehlende Werte und leere Texte
* Analyse der Textlängen
* Vergleich der Verteilungen zwischen den gewählten Kategorien

Erkenntnisse:

➡️ Starke Klassenungleichverteilung mit Dominanz von 5-Sterne-Bewertungen

➡️ Vorhandensein von Rezensionen ohne Textinhalt

➡️ Ähnliche Textlängenverteilungen in AllBeauty und HealthAndPersonalCare, kürzere Texte in HandmadeProducts

***

## 🛠️ Datenvorverarbeitung

Die Datenvorverarbeitung umfasste mehrere aufeinander abgestimmte Schritte:

### 🔧 1. Datenbereinigung & Selektion

* Entfernung nicht verifizierter Käufe
* Entfernung von Rezensionen mit sehr kurzen Texten (< 10 Zeichen)
* Reduktion des Datensatzes
* Beibehaltung ausschließlich relevanter Spalten (rating, title, text)

### 🧹 2. Textaufbereitung

* Zusammenführung von Rezensionstitel und Rezensionstext
* Umwandlung aller Zeichen in Kleinbuchstaben (Lowercasing)
* Tokenisierung
* Entfernung von Stoppwörtern
* Bereinigung von Sonderzeichen, Satzzeichen und Emojis
* Part-of-Speech-Tagging zur Vorbereitung der Lemmatisierung
* Lemmatisierung

### 🔄 3. Numerische Repräsentation

* Erstellung von TF-IDF-Vektoren zur Gewichtung relevanter Begriffe
* Nutzung eines N-Gram-Bereichs von (1,5) zur Erfassung kurzer Wortkombinationen und Negationen
* Speicherung der erzeugten TF-IDF-Vectorizers

***

## 🤖 Eingesetzte Modelle

#### **Logistische Regression**
 * Klassenbalancierung über `class_weight='balanced'`
 * Optimierung mit dem `saga`-Solver

#### **Naive Bayes**
 * Laplace-Glättung mit `α = 1`
 * Automatische Schätzung der Klassenprioren mit `class_prior=None`

***

## 📈 Evaluationsmetriken

Die Modellbewertung erfolgte anhand klassischer Metriken der Textklassifikation:

 * Accuracy
 * Precision
 * Recall
 * F1-Score
 * Confusion Matrix

Die **Confusion Matrix** wurde insbesondere genutzt, um Fehlklassifikationen zwischen benachbarten Sternebewertungen zu analysieren, was bei ordinalen Klassen besonders relevant ist.

***

## 🎯 Gesamtergebnisse

| Modell | Accuracy |
| :--- | :--- |
| **Logistische Regression** | ca. **52,9 %** |
| **Naive Bayes** | ca. **50,3 %** |

### Zentrale Beobachtungen:
 * Beide Modelle erreichten ähnliche Leistungswerte
 * Logistische Regression schnitt leicht besser ab als Naive Bayes
 * Extreme Bewertungen (1 und 5 Sterne) wurden zuverlässiger erkannt
 * Fehlklassifikationen traten überwiegend zwischen **benachbarten Klassen** auf (z. B. 4 ↔ 5)

Die vergleichsweise moderate Accuracy ist typisch für **multiklassige Sentimentanalyse**, da mit steigender Klassenzahl die Trennschärfe zwischen den Klassen abnimmt.

***

## 🧩 Herausforderungen und Limitationen

Unabhängig vom verwendeten Modell traten typische Probleme der Sentimentanalyse auf:

 * Negationen, Ironie und Sarkasmus
 * Polysemie und kontextabhängige Wortbedeutungen
 * Gemischte Sentiments innerhalb einzelner Rezensionen
 * Subjektive Bewertungslogik der Nutzer

Selbst leistungsfähige Modelle können keine korrekte Vorhersage treffen, wenn der vergebene Sternwert nicht konsistent zum Textinhalt ist (z. B. ein positiver Text mit niedriger Bewertung).
