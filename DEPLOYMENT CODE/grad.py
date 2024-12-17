import gradio as gr
import pickle
from sklearn.feature_extraction.text import CountVectorizer

def load_model():
    """
    Load the pre-trained model and vectorizer from pickle files.
    """
    model_path = "clf.pkl"
    vectorizer_path = "cv.pkl"
    
    try:
        # Load the model
        with open(model_path, "rb") as model_file:
            clf = pickle.load(model_file)
        
        # Load the vectorizer
        with open(vectorizer_path, "rb") as vectorizer_file:
            cv = pickle.load(vectorizer_file)
        
        return clf, cv
    except Exception as e:
        print(f"Error loading model or vectorizer: {e}")
        return None, None

def classify_text(clf, cv, test_data):
    """
    Classify the input text using the provided classifier and vectorizer.
    """
    try:
        # Vectorize the input text
        input_vectorized = cv.transform([test_data]).toarray()
        print(f"Vectorized input: {input_vectorized}")  # Debugging
        
        # Predict the label
        prediction = clf.predict(input_vectorized)[0]
        print(f"Raw prediction: {prediction}, Type: {type(prediction)}")  # Debugging
        
        # Define label mapping
        labels = {0: "hate speech detected", 1: "offensive speech", 2: "Nor hatred nor offensive"}
        
        # Handle unexpected outputs (e.g., string predictions)
        if isinstance(prediction, str):
            label_to_int = {"hate speech detected": 0, "offensive speech": 1, "Nor hatred nor offensive": 2}
            prediction = label_to_int.get(prediction, -1)
        
        return labels.get(prediction, "nor hatred nor offensive")
    except Exception as e:
        print(f"Error during prediction: {e}")
        return "Error in prediction. Please try again."

def main():
    """
    Main function to set up and launch the Gradio interface.
    """
    # Load the model and vectorizer
    clf, cv = load_model()
    
    if clf is None or cv is None:
        print("Failed to load model or vectorizer. Exiting.")
        return
    
    # Define the Gradio interface function
    def gradio_interface_fn(test_data):
        print(f"Received text: {test_data}")  # Debugging
        return classify_text(clf, cv, test_data)
    
    # Create the Gradio interface
    interface = gr.Interface(
        fn=gradio_interface_fn,
        inputs=gr.Textbox(lines=2, placeholder="Enter Text Here..."),
        outputs="text",
        title=" Social Media Toxicity Classification",
        description="Classify input text as 'Hateful', 'Offensive', or 'Neither' using a pre-trained machine learning model."
    )
    
    # Launch the Gradio app
    interface.launch()

if __name__ == "__main__":
    main()


