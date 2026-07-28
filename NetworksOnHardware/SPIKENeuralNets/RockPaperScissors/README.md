## Rock Paper Scissors Classifier
This classifier takes in three motor positions (fingers), and classifies the overall hand position as rock, paper, or scissors. 2 fingers down represented scissors, all fingers down represented paper, and all fingers up represented rock.

### Files
| Path | Purpose |
|---|---|
| `RPSClassifier.py` | Defines an RPSClassifier, defines structure and methods to train and pass fowards and backwards|
| `main.py` | Creates and trains an RPSClassifier, extracts the weights and biases, and puts them into a hub program to load onto the SPIKE |
| `rps_hub_program.py` | The hub program loaded onto the SPIKE built my main.py |