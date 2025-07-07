# Tool Fetching Domain Gym Environment
## Used as a scenario for representing Communication in Ad hoc Teamwork (CAT)
-------------------------------------------
## Installation
* Clone the repository
```bash
git clone https://github.com/williammacke/adhoc_qa_temp
cd adhoc_qa_temp
```
* Create Virtual Environment
```bash
python -m virtualenv env
```
* Activate Virtual Environment

* Linux
```bash
source env/bin/activate
```

* Windows
```bat
env/Scripts/activate
```
* Install Prerequisites
```bash
pip install -r requirements.txt
```
--------------------------------------------------
## Running the Demos
* To run a demonstration, simple run the desired demo in python
```bash
python demos/demo_name.py
```
------------------------------------------------
## Running Experiments
* To run an experiment, simple run the desired demo in python
```bash
python experiments/experiment_name.py
```
* To get information on flags, run the experiments help flag
```bash
python experiments/experiment_name.py --help
```
----------------------------------------------------
## Graphing Results
* To plot results, simple run graph file
```bash
python graphing/plot_results.py results_file_name
```
* To get information on flags, run the graphing help flag
```bash
python graphing/plot_results.py --help
```

