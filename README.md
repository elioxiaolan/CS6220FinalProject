# CS6220-Project

## Setup Instructions to run this app using Docker

1. **Clone the Repository:** Clone the repository:

    ```bash
    git clone [Repository URL]
    ```

2. **Navigate to the Repository Directory:** After cloning, navigate to the directory of the cloned repository:

    ```bash
    cd [Repository Name]
    ```

3. **Build the Docker Image:** Build the Docker image:

    ```bash
    docker build -t streamlit_app .
    ```

4. **Run the Docker Container:** Start the Docker container:

    ```bash
    docker run -p 8501:8501 streamlit_app
    ```

5. **Access the App:** Once the Docker container is running, open your web browser and go to http://localhost:8501 

## Additional Notes

- new_train2.csv: sample csv that could be used for Data Analysis and Performance Evaluation
- Test.csv: sample csv that could be used for Prediction
