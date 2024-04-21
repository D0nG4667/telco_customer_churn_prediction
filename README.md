# Telco_customer_churn_prediction

A machine learning project to predict customer churn, built with Python and scikit-learn, designed for telecom companies aiming to reduce customer churn and as a result improve customer retention.

## Overview

This repository contains data analysis, insights, and machine learning modelling for customer churn prediction.

## Key Objectives

- Analyze funding trends and dynamics within the Indian start-up ecosystem.
- Examine sector-wise distribution of funding and geographical trends.
- Identify prominent investors and their impact on start-up funding.
- Provide valuable insights to inform strategic decision-making for investors, entrepreneurs, and stakeholders.

## Framework

The CRoss Industry Standard Process for Data Mining (CRISP-DM).

## Features

- Jupyter Notebook containing data analysis, visualizations, and interpretation.
- Detailed documentation outlining methodology, data sources, and analysis results.
- Interactive visualizations in Power BI showcasing funding trends and key insights.

### PowerBI Dashboard

![Dashboard](/screenshots/dashboard.png)

## Technologies Used

- Anaconda
- Python
- Pandas
- NumPy
- Plotly
- Jupyter Notebooks
- Git
- Scipy
- Sklearn
- Xgboost
- Catboost
- Lightgbm
- Imblearn
- Pyodbc
- Re
- Typing

## Installation

### Quick install

```bash
 pip install -r requirements.txt
```

### Recommended install

```bash
conda env create -f churn_environment.yml
```

## Sample Code- used to generate the performance metric of a list of models

```python
def info(models: Union[ValuesView[Pipeline], List[Pipeline]], metric: Callable[..., float], **kwargs) -> List[Dict[str, Any]]:
    """
    Generates a list of dictionaries, each containing a model's name and a specified performance metric.

    Parameters:
    - models (List[Pipeline]): A list of model pipeline instances.
    - metric (Callable[..., float]): A function used to evaluate the model's performance. Expected to accept
      parameters like `y_true`, `y_pred`, and `average`, and return a float.
    - **kwargs: Additional keyword arguments to be passed to the metric function or any other function calls inside `info`. Can pass

    Returns:
    - List[Dict[str, Any]]: A list of dictionaries with model names and their evaluated metrics.
    """
    def get_metric(model, kwargs):
         
        # Add default kwargs for callable metric to kwargs. Consider is they are present in kwargs
        if 'X_train' and 'y_train_encoded' in kwargs:
            model.fit(kwargs[X_train], kwargs[y_train_encoded])
        else:
            # Fit final pipeline to training data            
            model.fit(X_train, y_train_encoded)
        
        if 'y_eval_encoded' in kwargs:
            kwargs['y_true'] = kwargs['y_eval_encoded']
        else:
            kwargs['y_true'] = y_eval_encoded
            
        if 'X_eval' in kwargs:
            kwargs['y_pred'] = model.predict(kwargs[X_eval])
        else:
            kwargs['y_pred'] = model.predict(X_eval)   
        
        # Sanitize the metric arguments, use only valid metric parameters
        kwargs = {k: value for k, value in kwargs.items() if k in inspect.signature(metric).parameters.keys()}
        
        return metric(**kwargs)    
    
    info_metric = [
        {
            'model_name': model['classifier'].__class__.__name__,
            f'Metric ({metric.__name__}_{kwargs['average'] if 'average' in kwargs else ''})': get_metric(model, kwargs),
        } for model in models
    ]

    return info_metric


```

## Contributions

### How to Contribute

1. Fork the repository and clone it to your local machine.
2. Explore the Jupyter Notebooks and documentation.
3. Implement enhancements, fix bugs, or propose new features.
4. Submit a pull request with your changes, ensuring clear descriptions and documentation.
5. Participate in discussions, provide feedback, and collaborate with the community.

## Feedback and Support

Feedback, suggestions, and contributions are welcome! Feel free to open an issue for bug reports, feature requests, or general inquiries. For additional support or questions, you can connect with me on [LinkedIn](https://www.linkedin.com/in/dr-gabriel-okundaye).

Link to article on Medium: [Telco Customer Churn Prediction: Unveiling Insights with Data Analysis and Machine Learning](https://medium.com/@gabriel007okuns/telco-customer-churn-prediction-unveiling-insights-with-data-analysis-and-machine-learning-9347a69b2dfe)

## Author

[Gabriel Okundaye](https://www.linkedin.com/in/dr-gabriel-okundaye).

## License

[MIT](/LICENSE)
