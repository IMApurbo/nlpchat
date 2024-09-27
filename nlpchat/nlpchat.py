import numpy as np
import pickle
import tensorflow as tf
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

class NlpChat:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.label_encoder = LabelEncoder()
        self.intents = []
        self.nn_model = None  # Placeholder for the neural network model

    def add_intent(self, tag, patterns, responses):
        self.intents.append({
            'tag': tag,
            'patterns': patterns,
            'responses': responses
        })

    def train(self):
        sentences = []
        labels = []

        for intent in self.intents:
            for pattern in intent['patterns']:
                sentences.append(pattern)
                labels.append(intent['tag'])

        # Encode the labels
        encoded_labels = self.label_encoder.fit_transform(labels)

        # Convert sentences to embeddings
        sentence_embeddings = self.model.encode(sentences)

        # Split the data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(sentence_embeddings, encoded_labels, test_size=0.2, random_state=42)

        # Define the Neural Network model
        self.nn_model = tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=(sentence_embeddings.shape[1],)),  # Input layer
            tf.keras.layers.Dense(32, activation='relu'),  # Hidden layer
            tf.keras.layers.Dense(16, activation='relu'),  # Additional hidden layer
            tf.keras.layers.Dense(len(self.label_encoder.classes_), activation='softmax')  # Output layer
        ])

        # Compile the model
        self.nn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        # Train the model
        self.nn_model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test))

        # Evaluate the model
        test_loss, test_acc = self.nn_model.evaluate(X_test, y_test)
        print(f"Training completed. Test accuracy: {test_acc * 100:.2f}%")

    def get_response(self, user_input):
        # Encode the new sentence
        embedding = self.model.encode([user_input])
        
        # Predict the intent
        predicted_label = np.argmax(self.nn_model.predict(embedding), axis=1)
        predicted_intent = self.label_encoder.inverse_transform(predicted_label)[0]

        # Find the corresponding response
        for intent in self.intents:
            if intent['tag'] == predicted_intent:
                return np.random.choice(intent['responses'])

    def get_intent(self, user_input):
        # Encode the new sentence
        embedding = self.model.encode([user_input])
        
        # Predict the intent
        predicted_label = np.argmax(self.nn_model.predict(embedding), axis=1)
        predicted_intent = self.label_encoder.inverse_transform(predicted_label)[0]
        
        return predicted_intent

    def save_model(self, model_filename, encoder_filename):
        self.nn_model.save(model_filename)  # Save the Keras model
        with open(encoder_filename, 'wb') as f:  # Save the label encoder
            pickle.dump(self.label_encoder, f)

    def load_model(self, model_filename, encoder_filename):
        self.nn_model = tf.keras.models.load_model(model_filename)  # Load the Keras model
        with open(encoder_filename, 'rb') as f:  # Load the label encoder
            self.label_encoder = pickle.load(f)
