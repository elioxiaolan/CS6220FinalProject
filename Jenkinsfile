pipeline {
    agent any

    stages {
        stage('Checkout') {
            steps {
                // Checkout code from SCM
                branches: [[name: '*/master']],
                checkout scm
            }
        }

        stage('Build Docker Image') {
            steps {
                script {
                    // Build the Docker image using the Dockerfile in the repository
                    docker.build("streamlit_app:${env.BUILD_ID}")
                }
            }
        }

        stage('Run Docker Container') {
            steps {
                script {
                    // Run the Docker container in detached mode
                    sh 'docker run -d -p 8501:8501 --name streamlit_app_${env.BUILD_ID} streamlit_app:${env.BUILD_ID}'
                }
            }
        }

        stage('Test Application') {
            steps {
                script {
                    // (Optional) Add steps to verify the application is running correctly
                    sh 'curl -I http://localhost:8501'
                }
            }
        }
    }

    post {
        always {
            // Clean up: Stop and remove the Docker container
            sh 'docker stop streamlit_app_${env.BUILD_ID} || true'
            sh 'docker rm streamlit_app_${env.BUILD_ID} || true'
            // Optionally clean up the Docker image
            sh 'docker rmi streamlit_app:${env.BUILD_ID} || true'
            cleanWs() // Clean workspace after build
        }

        success {
            echo 'Docker image built, container started, and app is accessible!'
        }

        failure {
            echo 'Something went wrong with the build, container, or app access.'
        }
    }
}

