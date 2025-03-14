# Welcome to the KLMS Tool version of AvengER

AvengER is a tool for ensembling and fine-tuning LLMs for SELECT Prompts in Entity Resolution. It was published in XXXXX and the original repo can be found [here](https://github.com/alexZeakis/AvengER).

# Instructions

## Input
AvengER can be executed from the KLMS with the following input json:

```json
{
	"input": {
		"left_file": [
		    "XXXXXXXX-bucket/temp1.csv"
		],
		"right_file": [
			"XXXXXXXX-bucket/temp2.json"
		],
		"sample_file": [
			"XXXXXXXX-bucket/sample.csv"
		],
		"ground_truth_file": [
			"XXXXXXXX-bucket/gt.csv"
		],		
	},
	"output": {
		"prompts": "/path/to/write/the/file",
		"responses": "/path/to/write/the/file"
    },
	"parameters": {
        "seed": 1924,
        "serialization": "DITTO",
        "task_description": "SELECT",
        "model": "llama3-8b-8192",
        "examples": "None",
        "experts": "None",
        "reverse": False,
        "blocking_k": 10,
        "blocking_method": "intersection",
        "device": "cpu"
    },
    "secrets": {
        "endpoint": "Endpoint/of/LLM"
		"token": "Token/for/Endpoint"
	},
	"minio": {
		"endpoint_url": "minio.XXXXXX.gr",
		"id": "XXXXXXXX",
		"key": "XXXXXXXX",
		"skey": "XXXXXXXX"
    }
}
```

### Input

- **`left_file`** *(str, required)*  
  Path to the left file containing one side of the entity pairs.  

- **`right_file`** *(str, required)*  
  Path to the right file containing the other side of the entity pairs.  

- **`sample_file`** *(str, optional)*  
  Path to a sample file for example selection or matching. If not provided, blocking with `k=10` will be executed.  
  
- **`ground_truth_file`** *(str, optional)*    
  Path to the file containing the true_pairs.

### Parameters  

- **`seed`** *(int, optional, default=1924)*  
  Seed for shuffling operations to ensure reproducibility.  

- **`serialization`** *(str, required)*  
  Defines how input data is serialized. Must be one of:  
  - `"DITTO"`  
  - `"schema_agnostic"`  
  - `"llm"`  

- **`task_description`** *(str, required)*  
  Specifies the type of task description used in the prompt. Options include:  
  - `"SELECT"`  
  - `"EXPLAIN"`  
  - `"CONFIDENCE"`  
  - `"MATCH"`  
  - `"COMPARE"`  

- **`model`** *(str, required)*  
  The language model to use for processing (e.g., `"llama3-8b-8192"`).  

- **`examples`** *(str, optional, default="None")*  
  Strategy for selecting examples. Available options:  
  - `"SELECT"`  
  - `"MATCH"`  

- **`experts`** *(str, optional, default="None")*  
  Path to a directory containing expert answers to be included in the prompt.  

- **`reverse`** *(bool, optional, default=False)*  
  Whether to generate a reverse prompt. Applies only to `"MATCH"` and `"COMPARE"` task descriptions.  

- **`blocking_k`** *(int, optional, default=10)*  
  Number of nearest neighbors to consider during blocking. Used when no `sample_file` is provided.  
  
- **`blocking_method`** *(str, optional, default="intersection")*  
  Strategy used for blocking. Options include:  
  - `"left_to_right"` – Considers only candidates where entities from the left file are matched to the right.  
  - `"right_to_left"` – Considers only candidates where entities from the right file are matched to the left.  
  - `"intersection"` (default) – Retains entity pairs appearing in multiple blocking methods.  
  - `"union"` – Combines all candidate pairs from different blocking methods.  

- **`device`** *(str, optional, default="cpu")*  
  Specifies the computing device to use for model inference. Options include:  
  - `"cpu"` – Runs the model on the CPU.  
  - `"cuda"` – Uses a GPU for faster processing (if available).  



## Output

The output of AvengER has the following format:

```json
{
    "message": "Tool executed successfully!",
	"output": {
		"prompts": "path_of_prompts_file",
		"responses": "path_of_responses_file"			
    }
	"metrics": {	
        "recall": 1.0,
        "precision": 1.0,
        "f1": 1.0,
        "time": 1.698,
        "size": 70.0,
        "vect_time": 4.447, 
        "blocking_time": 0.615, 
        "blocking_precision": 0.15,
        "blocking_recall": 0.95,
        "blocking_f1": 0.26 
    },
	"status": "success"
}
```
