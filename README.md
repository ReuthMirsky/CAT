# CAT: Communication in Ad Hoc Teamwork

**CAT (Communication in Ad Hoc Teamwork)** is a framework for studying and developing agents capable of effective collaboration in dynamic, ad hoc teams. This environment introduces the **Tool Fetching Domain**, a testbed specifically designed to explore explicit communication strategies in teamwork scenarios where agents may not share prior coordination or knowledge.

## 🧠 Overview

CAT provides a variety of algorithms and tools for experimenting with explicit communication between agents in multi-agent reinforcement learning settings. The framework emphasizes sequential decision-making, limited inquiries, and the value of communication in collaborative environments.

The environment and research in this project are grounded in the following key papers:

- Mirsky, R., Macke, W., Wang, A., Yedidsion, H., & Stone, P. (2020). *A penny for your thoughts: The value of communication in ad hoc teamwork*. International Joint Conference on Artificial Intelligence.
- Macke, W., Mirsky, R., & Stone, P. (2020). *Query Content in Sequential One-shot Multi-Agent Limited Inquiries when Communicating in Ad Hoc Teamwork*. Workshop on Distributed and Multi-Agent Planning (DMAP), ICAPS.
- Macke, W., Mirsky, R., & Stone, P. (2021). *Expected value of communication for planning in ad hoc teamwork*. Proceedings of the AAAI Conference on Artificial Intelligence.
- Suriadinata, J., Macke, W., Mirsky, R., & Stone, P. (2021). *Reasoning about human behavior in ad hoc teamwork*. Adaptive and Learning Agents Workshop, AAMAS.

## 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/williammacke/adhoc_qa_temp
cd adhoc_qa_temp
```

Create and activate a virtual environment:

**Linux:**
```bash
python -m virtualenv env
source env/bin/activate
```

**Windows:**
```bat
python -m virtualenv env
env\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## 🧪 Running the Demos

To run a demonstration:

```bash
python demos/demo_name.py
```

## 📊 Running Experiments

To run an experiment:

```bash
python experiments/experiment_name.py
```

For information on available flags:

```bash
python experiments/experiment_name.py --help
```

## 📈 Graphing Results

To plot results:

```bash
python graphing/plot_results.py results_file_name
```

For help with plotting options:

```bash
python graphing/plot_results.py --help
```
