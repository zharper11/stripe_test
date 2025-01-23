from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
import numpy as np
from openai import OpenAI
import pandas as pd
import tiktoken
import json
import math
import asyncio
import os
from scipy.spatial.distance import cdist
from datetime import datetime, timedelta
import stripe
from azure.cosmos import CosmosClient, PartitionKey
import sys
from parallel_api import process_json

# Configuration
app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.INFO)
COSMOS_URI = os.getenv("COSMOS_URI")
COSMOS_KEY = os.getenv("COSMOS_KEY")
DATABASE_NAME = os.getenv("DATABASE_NAME")
CONTAINER_NAME = os.getenv("CONTAINER_NAME")
client = CosmosClient(COSMOS_URI, credential=COSMOS_KEY)
database = client.create_database_if_not_exists(DATABASE_NAME)
container = database.create_container_if_not_exists(
    id=CONTAINER_NAME,
    partition_key=PartitionKey(path="/id"),
    default_ttl=None
)
async def update_user_metrics(container, user_id, tokens_used=0, training_run=False, module_type=None, input_size=None, categories_size=None):
    """
    Update user metrics in Cosmos DB with session history
    """
    try:
        if not user_id:
            raise ValueError("user_id cannot be null or empty")
            
        # Query existing user document
        query = f"SELECT * FROM c WHERE c.userId = '{user_id}'"
        items = list(container.query_items(query=query, enable_cross_partition_query=True))
        
        current_time = datetime.utcnow().isoformat()
        
        # Create session entry
        session_entry = {
            'timestamp': current_time,
            'tokens_used': tokens_used,
            'training_run': training_run
        }
        
        # Add module-specific data
        if module_type:
            session_entry['module_type'] = module_type
            session_entry['input_size'] = input_size
            if module_type == 'classification' and categories_size is not None:
                session_entry['categories_size'] = categories_size
        
        if items:
            # Get existing document
            doc = items[0]
            
            # Initialize metrics if they don't exist
            if 'metrics' not in doc:
                doc['metrics'] = {
                    'total_tokens': 0,
                    'training_runs': 0,
                    'sessions': [],
                    'last_updated': current_time
                }
            
            # Update general metrics
            doc['metrics']['total_tokens'] += tokens_used
            if training_run:
                doc['metrics']['training_runs'] = doc['metrics'].get('training_runs', 0) + 1
            
            # Add new session to sessions array
            if 'sessions' not in doc['metrics']:
                doc['metrics']['sessions'] = []
            doc['metrics']['sessions'].append(session_entry)
            
            # Update last_updated timestamp
            doc['metrics']['last_updated'] = current_time
            
            # Update document in Cosmos DB
            container.upsert_item(doc)
            
        else:
            # Create new document for new user
            new_doc = {
                "id": user_id,
                "userId": user_id,
                "metrics": {
                    'total_tokens': tokens_used,
                    'training_runs': 1 if training_run else 0,
                    'sessions': [session_entry],
                    'last_updated': current_time
                }
            }
            container.create_item(new_doc)
            
        return True
    except ValueError as ve:
        logging.error(f"Validation error in update_user_metrics: {ve}")
        return False
    except Exception as e:
        logging.error(f"Error updating user metrics: {e}")
        return False

# Save API Key
@app.route('/api/save_key', methods=['POST'])
def save_api_key():
    data = request.get_json()
    user_id = data.get('userId')
    api_key = data.get('apiKey')

    if not user_id or not api_key:
        return jsonify({"error": "userId and apiKey are required"}), 400

    try:
        # Validate API key format
        if not api_key.startswith('sk-') or len(api_key) < 20:
            return jsonify({"error": "Invalid API key format"}), 400

        # First, get existing document if it exists
        query = f"SELECT * FROM c WHERE c.userId = '{user_id}'"
        items = list(container.query_items(query=query, enable_cross_partition_query=True))
        
        if items:
            # Update existing document
            doc = items[0]
            doc['apiKey'] = api_key
            doc['lastUpdated'] = datetime.utcnow().isoformat()
            container.upsert_item(doc)
        else:
            # Create new document
            container.upsert_item({
                "id": user_id,
                "userId": user_id,
                "apiKey": api_key,
                "lastUpdated": datetime.utcnow().isoformat(),
                "metrics": {
                    'total_tokens': 0,
                    'training_runs': 0,
                    'sessions': [],
                    'last_updated': datetime.utcnow().isoformat()
                }
            })
            
        return jsonify({"message": "API key saved successfully"}), 200

    except Exception as e:
        logging.error(f"Error saving API key: {e}")
        return jsonify({"error": str(e)}), 500
@app.route('/api/load_key', methods=['GET'])
def load_api_key():
    user_id = request.args.get('userId')

    if not user_id:
        return jsonify({"error": "userId is required"}), 400

    try:
        query = f"SELECT c.apiKey FROM c WHERE c.userId = '{user_id}'"
        items = list(container.query_items(
            query=query,
            enable_cross_partition_query=True
        ))

        if not items:
            return jsonify({"error": "API key not found"}), 404

        # Validate stored API key format
        api_key = items[0].get('apiKey', '')
        if not api_key.startswith('sk-') or len(api_key) < 20:
            return jsonify({"error": "Stored API key is invalid"}), 400

        return jsonify({"apiKey": api_key}), 200

    except Exception as e:
        logging.error(f"Error loading API key: {e}")
        return jsonify({"error": str(e)}), 500
gptmodel = "gpt-4o"
request_url= "https://api.openai.com/v1/chat/completions"
max_requests_per_minute =250
max_tokens_per_minute = 30000
token_encoding_name = "cl100k_base"
max_attempts = 5
logging_level = 20
stripe.api_key = os.getenv("STRIPE_KEY")

# Add these new routes to your existing app.py
@app.route('/api/create-checkout-session', methods=['POST'])
def create_checkout_session():
    try:
        data = request.get_json()
        user_id = data.get('userId')

        if not user_id:
            return jsonify({"error": "User ID is required"}), 400

        checkout_session = stripe.checkout.Session.create(
            payment_method_types=['card'],
            line_items=[{
                'price': 'price_1Qjkn0ECtCMmOAVYVtQmrYtC',  # Replace with your Stripe price ID
                'quantity': 1,
            }],
            mode='subscription',
            success_url='https://localhost:3000/taskpane.html?payment=success',
            cancel_url = 'https://localhost:3000/taskpane.html?payment=canceled',
            client_reference_id=user_id,
        )

        return jsonify({'sessionId': checkout_session.id}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/webhook', methods=['POST'])
def webhook():
    payload = request.get_data()
    sig_header = request.headers.get('Stripe-Signature')

    try:
        event = stripe.Webhook.construct_event(
            payload, sig_header, os.getenv("WEBHOOK_SECRET")
        )

        if event['type'] == 'checkout.session.completed':
            session = event['data']['object']
            user_id = session['client_reference_id']
            subscription_id = session['subscription']

            # Update subscription in Cosmos DB
            query = f"SELECT * FROM c WHERE c.userId = '{user_id}'"
            items = list(container.query_items(query=query, enable_cross_partition_query=True))
            
            if items:
                doc = items[0]
                doc['subscription'] = {
                    'active': True,
                    'subscriptionId': subscription_id,
                    'startDate': datetime.utcnow().isoformat(),
                    'endDate': (datetime.utcnow() + timedelta(days=365)).isoformat()
                }
                container.upsert_item(doc)

        return jsonify({'status': 'success'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 400
def verify_subscription(user_id):
    try:
        query = f"SELECT * FROM c WHERE c.userId = '{user_id}'"
        items = list(container.query_items(query=query, enable_cross_partition_query=True))
        
        if not items:
            return False

        doc = items[0]
        subscription = doc.get('subscription', {})
        
        if not subscription or not subscription.get('active'):
            return False

        # Check if subscription is expired
        end_date = datetime.fromisoformat(subscription['endDate'])
        if end_date < datetime.utcnow():
            # Update subscription status to inactive
            doc['subscription']['active'] = False
            container.upsert_item(doc)
            return False

        return True
    except Exception as e:
        logging.error(f"Error verifying subscription: {e}")
        return False

@app.route('/api/uniqueness', methods=['POST'])
def analyze_uniqueness():
    try:
        logging.info("Uniqueness analysis request received")
        data = request.get_json()

        # Extract API parameters from the request payload
        api_key = data.get('apiKey')
        user_id = data.get('userId')
        if not verify_subscription(user_id):
            return jsonify({"error": "Active subscription required"}), 403
        if not api_key:
            logging.error("API key is missing.")
            return jsonify({"error": "API key is required."}), 400
        # Initialize OpenAI client
        client = OpenAI(api_key=api_key)
        
        # Validate input data
        input_data = data.get('inputData', [])
        if not input_data:
            logging.error("Input data is empty.")
            return jsonify({"error": "Input data cannot be empty."}), 400

        # Create DataFrame from input
        df_item = pd.DataFrame([row[0] for row in input_data], columns=['Item'])
        
        # Generate embeddings
        async def process_embeddings_completion(client, df_item):
            tasks = []
            for index, row in df_item.iterrows():
                item_input = row['Item']
                tasks.append(
                    asyncio.to_thread(
                        client.embeddings.create,
                        input=item_input,
                        model="text-embedding-3-small",
                        dimensions=200,
                    )
                )
            return await asyncio.gather(*tasks, return_exceptions=True)

        # Run embedding generation
        responses = asyncio.run(process_embeddings_completion(client, df_item))
        # Extract embeddings from responses
        embeddings = []
        for response in responses:
            if isinstance(response, Exception):
                logging.error(f"Error generating embedding: {response}")
                embeddings.append(np.zeros(200))  # Fallback for errors
            else:
                embeddings.append(response.data[0].embedding)

        # Convert embeddings to numpy array
        embeddings_matrix = np.vstack(embeddings)

        # Calculate Euclidean distances
        euclidean_distances = cdist(embeddings_matrix, embeddings_matrix, metric='euclidean') # type: ignore
        np.fill_diagonal(euclidean_distances, np.nan)

        # Calculate statistical measures
        overall_mean = np.nanmean(euclidean_distances)
        overall_min = np.nanmin(euclidean_distances)

        # Store distances in DataFrame
        df_item['euclidean_distances'] = [row.tolist() for row in euclidean_distances]
        df_item['mean_distance'] = df_item['euclidean_distances'].apply(lambda x: np.nanmean(x))
        df_item['mean_distance_std'] = df_item['euclidean_distances'].apply(lambda x: np.nanstd(x))

        # Calculate LHS standard deviation
        def lhs_std(distances, mean_distance):
            lhs_differences = [mean_distance - d for d in distances if d < mean_distance and not np.isnan(d)]
            return np.mean(lhs_differences) if lhs_differences else np.nan

        df_item['LHS_std'] = df_item.apply(
            lambda row: lhs_std(row['euclidean_distances'], row['mean_distance']),
            axis=1
        )

        # Calculate z-scores
        df_item['mean_z_score'] = (df_item['mean_distance'] - overall_mean) / df_item['mean_distance_std']
        df_item['zero_z_score'] = (-df_item['mean_distance'] - overall_min) / df_item['LHS_std']

        # Calculate final uniqueness score
        df_item['unique_score'] = df_item['zero_z_score'] - df_item['mean_z_score']

        # Normalize scores to 0-1 range
        score_min = df_item['unique_score'].min()
        score_max = df_item['unique_score'].max()
        df_item['unique_score'] = (df_item['unique_score'] - score_min) / (score_max - score_min)

        # Format results to match original output structure
        uniqueness_scores = [[float(score)] for score in df_item['unique_score']]
        embedding_tokens = len(input_data) * 200
        asyncio.run(update_user_metrics(
            container=container,
            user_id=user_id,
            tokens_used=embedding_tokens,
            module_type='uniqueness',
            input_size=len(input_data)
        ))
        logging.info("Uniqueness analysis complete")
        return jsonify(uniqueness_scores), 200

    except Exception as e:
        logging.error(f"Unexpected error in uniqueness analysis: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/analyze', methods=['POST'])
def analyze():
    try:
        logging.info("Analysis request received")
        data = request.get_json()
        #logging.info(f"Received full data payload: {data}")
        # Extract API parameters from the request payload
        api_key = data.get('apiKey')
        user_id = data.get('userId')
        if not verify_subscription(user_id):
            return jsonify({"error": "Active subscription required"}), 403
        if not api_key:
            logging.error("API key is missing.")
            return jsonify({"error": "API key is required."}), 400
        
        # Validate input data
        item_data = data.get('inputData', [])
        category_list = data.get('categories', [])
        instructions = data.get('instructions', '')
        rerun_with_training = bool(data.get('rerun_with_training', ''))
        input_size = len(item_data)
        categories_size = len(category_list)
        
        

        if not item_data or all(not item for sublist in item_data for item in sublist):
            logging.error("Input data is empty.")
            return jsonify({"error": "Input data cannot be empty."}), 400

        if not category_list or all(not category for category in category_list):
            logging.error("Categories are empty.")
            return jsonify({"error": "Categories cannot be empty."}), 400
        if not instructions:
            instructions = "No additional instructions provided by user"

        # Convert validated lists to DataFrames
        df_item = pd.DataFrame(item_data, columns=['Item'])
        df_category = pd.DataFrame(category_list, columns=['Category'])

        df_item['Response'] = ""
        df_item['Confidence'] = ""

        # Load text file content (assumed path needs to be specified)
        #txt_file_path = r'C:\ChatNoir\analysis\Prompt.txt'  # Update this path
        #with open(txt_file_path, 'r') as file:
        txt_content = """You are a data scientist data assistant and your job is to sort [Item] into one of the following categories: [Categories]

The additional context has been provided to help complete the task:
[Instructions]
You are only to return one of the categories and no other response. Please provide your best guess when there is no certain choice. Final response has to be from given categories [Categories]
"""

        # Define the weighting value
        weighting = 5

        # Initialize an empty dictionary to store the weighted tokens for all categories
        tokens_list = {}
        category_token_counts = {}
        df_category['tokens'] = None
        
        # Loop through categories to tokenize
        encoding = tiktoken.encoding_for_model(gptmodel)
        for index, row in df_category.iterrows():
            category = row[df_category.columns[0]]  # Assuming the category is in the first column
            
            # Tokenize the category using the encoding
            token_ids = tiktoken.encoding_for_model(gptmodel).encode(category)
            
            # Decode the token IDs back to tokens
            tokens = [tiktoken.encoding_for_model(gptmodel).decode([token_id]) for token_id in token_ids]
            
            # Add each token ID with its weighting to the tokens_list dictionary
            for token_id in token_ids:
                tokens_list[token_id] = weighting
            
            # Store the number of tokens for this category
            category_token_counts[category] = len(token_ids)
            
            # Add the tokens (as actual strings) to the DataFrame
            df_category.at[index, 'tokens'] = tokens

        # Find the category with the most tokens
        max_tokens_category = max(category_token_counts, key=category_token_counts.get)
        max_tokens_count = category_token_counts[max_tokens_category]

        # Function to calculate confidence score
        def calculate_confidence(logprobs_content, df_category, response_text):
            # Initialize arrays to store summed probabilities for each category per position
            category_sums = {
                'Selected category': [],
                'Not-selected category': [],
                'Model deviation': [],
                'Selected category- Incorrect tokens': []
            }
            # Find the matching category row index directly
            category_row_index = df_category[df_category['Category'] == response_text].index
            category_row = category_row_index[0] if len(category_row_index) > 0 else None

            # Loop through the object and categorize TopLogprob tokens
            tokens_set = {t.lower() for tokens in df_category['tokens'] for t in tokens}
            response_text_lower = response_text.lower()

            for item in logprobs_content:
                # Store the probabilities for each category at each token position
                token_probs = {key: 0.0 for key in category_sums}

                for top_logprob in item['top_logprobs']:
                    token_lower = top_logprob['token'].lower()
                    all_tokens_lower = [t.lower() for tokens in df_category['tokens'] for t in tokens]

                    probability = math.exp(top_logprob['logprob'])

                    if category_row is not None and token_lower in [t.lower() for t in df_category.at[category_row, 'tokens']]:
                        token_probs['Selected category'] += probability
                    elif token_lower not in tokens_set:
                        token_probs['Model deviation'] += probability
                    elif token_lower in response_text_lower:
                        token_probs['Selected category- Incorrect tokens'] += probability
                    else:
                        token_probs['Not-selected category'] += probability

                # Append the summed probabilities for each token position across all categories
                for category, prob in token_probs.items():
                    category_sums[category].append(prob)

            # Ensure all probability sum lists are the same length by padding with zeros
            max_length = len(category_sums['Selected category'])
            for category in category_sums:
                category_sums[category] += [0.0] * (max_length - len(category_sums[category]))

            # Create a summary DataFrame for the total probabilities at each position
            summary_df = pd.DataFrame({
                'Category': list(category_sums.keys()),
                **{f'Position {i+1}': [category_sums[category][i] for category in category_sums] for i in range(max_length)}
            })

            # Calculate weighting for Model Deviation
            total_model_deviation = 0
            for i in range(max_length):
                total_model_deviation += (1 - total_model_deviation) * (summary_df.at[summary_df[summary_df['Category'] == 'Model deviation'].index[0], f'Position {i + 1}'])

            # Calculate entropy probabilities using log approach to avoid numerical underflow
            entropy_probs = []
            for i in range(max_length + 1):
                if i < max_length:
                    # Start with log of Secondary Prediction at current position
                    log_probability = math.log(summary_df.at[summary_df[summary_df['Category'] == 'Not-selected category'].index[0], f'Position {i + 1}'] + 1e-10)
                    # Add log of all previous Primary Predictions, avoiding zero propagation
                    for j in range(i):
                        primary_prob = summary_df.at[summary_df[summary_df['Category'] == 'Selected category'].index[0], f'Position {j + 1}']
                        log_probability += math.log(primary_prob + 1e-10)
                else:
                    # For the final step, just use the log product of all Primary Predictions
                    log_probability = sum([math.log(summary_df.at[summary_df[summary_df['Category'] == 'Selected category'].index[0], f'Position {j + 1}'] + 1e-10) for j in range(max_length)])
                entropy_probs.append(math.exp(log_probability))

            # Normalize the entropy probabilities to ensure they sum to 1
            total_prob_sum = sum(entropy_probs)
            normalized_entropy_probs = [p / total_prob_sum for p in entropy_probs] if total_prob_sum > 0 else [1 / len(entropy_probs)] * len(entropy_probs)

            # Calculate entropy
            entropy = -sum([p * math.log2(p) for p in normalized_entropy_probs if p > 0])

            # Calculate maximum entropy (log2 of number of combinations)
            max_entropy = math.log2(max_length + 1)

            # Calculate total confidence based on entropy
            total_confidence = (1 - total_model_deviation) * (1 - entropy / max_entropy) if max_entropy > 0 else (1 - total_model_deviation)
            
            return total_confidence
        
                # Functions to run prompts
        def first_run(row, df_item, df_category):
            item = row[df_item.columns[0]]  # Assuming the first column is the 'Item' column
            categories = ", ".join(df_category[df_category.columns[0]].tolist())  # Assuming the first column is the 'Category' column
            filled_prompt = txt_content.replace("[Item]", item).replace("[Categories]", categories).replace("[Instructions]", instructions)
            return filled_prompt

        Retrain_Prompt_With_Corrections = """You are a data scientist data assistant and your job is to sort [Item] into one of the following categories: [Categories]

Here a few Examples:
[TRAINING_EXAMPLES]

Additional Context:
[Instructions]

Based on the examples above and the available categories, please classify the item.
Your response should ONLY contain the category name from the available categories listed above."""

        def retrain_run_with_corrections(row, df_item, df_category, corrections_data, instructions):
            item = row[df_item.columns[0]]
            categories = ", ".join(df_category[df_category.columns[0]].tolist())
            
            # Format training examples in a clear, numbered list
            training_examples = []
            for i, corr in enumerate(corrections_data, 1):
                training_examples.append(f"Example {i}:")
                training_examples.append(f"• Input: {corr[0]}")
                training_examples.append(f"• Correct Category: {corr[1]}")
                training_examples.append("")  # Add blank line between examples
            
            # Join all examples with newlines
            formatted_examples = "\n".join(training_examples).strip()
            
            filled_prompt = (Retrain_Prompt_With_Corrections
                .replace("[TRAINING_EXAMPLES]", formatted_examples)
                .replace("[Item]", item)
                .replace("[Categories]", categories)
                .replace("[Instructions]", instructions)
            )
            return filled_prompt

        def generate_json_objects(df_item, df_category, gptmodel, tokens_list, max_tokens_count):
            json_df_item = []
            for index, row in df_item.iterrows():
                filled_prompt = first_run(row, df_item, df_category) if not rerun_with_training else retrain_run_with_corrections(row, df_item, df_category, corrections_data, instructions)

                json_df_item_row = {
                    "model": gptmodel,
                    "logprobs": True,
                    "top_logprobs": 10,
                    "logit_bias": tokens_list,
                    "messages": [
                        {"role": "system", "content": "You are a data science tool used to help categorize information. Your answers will be fed into datatables."},
                        {"role": "user", "content": filled_prompt}
                    ],
                    "max_tokens": max_tokens_count,
                    "temperature": 0.50,
                    "metadata": {"row_id": index}
                }

                json_df_item.append(json_df_item_row)
                json.dumps(json_df_item)
            return json_df_item
        corrections_data = data.get('corrections', []) if rerun_with_training else []
        json_df_item = []
        json_df_item = generate_json_objects(df_item, df_category, gptmodel, tokens_list, max_tokens_count)     
        # Call parallel processor
        json_df_item = process_json(
            request_json=json_df_item,
            request_url=request_url,
            api_key=api_key,
            max_requests_per_minute=max_requests_per_minute,
            max_tokens_per_minute=max_tokens_per_minute,
            token_encoding_name=token_encoding_name,
            max_attempts=max_attempts,
            logging_level=logging_level,
            )
        total_tokens = 0
        for response in json_df_item:
            if not isinstance(response, Exception):
                response_data = response[1]
                # Get usage from the response if available
                if 'usage' in response_data:
                    total_tokens += response_data['usage']['total_tokens']
                else:
                    # If usage not available, use the max_tokens from request
                    request_data = response[0]  # Original request
                    total_tokens += request_data.get('max_tokens', 0)
        
        # Update metrics in Cosmos DB
        asyncio.run(update_user_metrics(
            container=container,
            user_id=user_id,
            tokens_used=total_tokens,
            training_run=rerun_with_training,
            module_type='classification',
            input_size=input_size,
            categories_size=categories_size
        ))
        

        for index, response in enumerate(json_df_item):
            if isinstance(response, Exception):
                print(f"Exception occurred in response for index {index}: {response}")
                df_item.at[index, 'Response'] = f"Error: {response}"  # Optionally log the error in the Response column
                df_item.at[index, 'Confidence'] = None
                continue
            # Extract response text and logprobs content
            #print(response)
            response_data = response[1]
            # Extract response text and logprobs content
            response_text = response_data['choices'][0]['message']['content']
            logprobs_content = response_data['choices'][0]['logprobs']['content']
            row_id = response[2]['row_id']
            # Calculate confidence score if logprobs are available
            confidence_score = None
            if logprobs_content:
                confidence_score = calculate_confidence(logprobs_content, df_category, response_text)

            # Handle missing categories in df_category
            if response_text not in df_category['Category'].tolist():
                response_text = "Error: Response not in original categories: " + response_text

            # Update df_item at the corresponding index with the response and confidence score
            df_item.at[row_id, 'Response'] = response_text
            df_item.at[row_id, 'Confidence'] = confidence_score
        logging.info("Analysis complete")

        #print(df_item)
        return jsonify(df_item[['Item', 'Response', 'Confidence']].to_dict(orient='records')), 200
        
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        return jsonify({"error": str(e)}), 500
@app.route('/api/open_format', methods=['POST'])
def open_format():
    try:
        logging.info("Open format request received")
        data = request.get_json()

        # Extract API parameters from the request payload
        api_key = data.get('apiKey')
        user_id= data.get('userId')
        if not verify_subscription(user_id):
            return jsonify({"error": "Active subscription required"}), 403
        if not api_key:
            logging.error("API key is missing.")
            return jsonify({"error": "API key is required."}), 400
        
        # Initialize OpenAI client
        client = OpenAI(api_key=api_key)
        
        # Validate input data
        input_data = data.get('inputData', [])
        instructions = data.get('instructions', '')
        model = data.get('model')
        temperature = float(data.get('temperature'))

        if not input_data:
            logging.error("Input data is empty.")
            return jsonify({"error": "Input data cannot be empty."}), 400

        if not instructions:
            logging.error("Instructions are empty.")
            return jsonify({"error": "Instructions cannot be empty."}), 400

        # Create DataFrame from input
        df_item = pd.DataFrame([row[0] for row in input_data], columns=['Input'])
        df_item['Response'] = ""

        # Create prompt template
        prompt_template = """You are my assistant and your job is to help with the following task. Please be concise and to the point in your response. Like I would be more then happy if you give answer in one word

Task: {instructions}

Input: {input_text}

Please provide your response in a direct and concise manner. Your main goal is to give as to the point answer as possible"""

        async def process_completions(client, df_item, instructions, model, temperature):
            tasks = []
            for index, row in df_item.iterrows():
                prompt = prompt_template.format(
                    instructions=instructions,
                    input_text=row['Input']
                )
                
                tasks.append(
                    asyncio.to_thread(
                        client.chat.completions.create,
                        model=model,
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant that provides concise, direct responses."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=temperature
                    )
                )
            return await asyncio.gather(*tasks, return_exceptions=True)

        # Run completion generation using asyncio.run
        responses = asyncio.run(process_completions(client, df_item, instructions, model, temperature))

        # Process responses and format results
        result_data = []
        for idx, response in enumerate(responses):
            if isinstance(response, Exception):
                logging.error(f"Error in processing item {idx}: {response}")
                result_data.append([input_data[idx][0], f"Error: {str(response)}"])
                continue
                
            response_text = response.choices[0].message.content
            result_data.append([input_data[idx][0], response_text])
        total_tokens = sum(
            response.usage.total_tokens 
            for response in responses 
            if not isinstance(response, Exception)
        )
        asyncio.run(update_user_metrics(
            container=container,
            user_id=user_id,
            tokens_used=total_tokens,
            module_type='open_format',
            input_size=len(input_data)
        ))
        logging.info("Open format analysis complete")
        return jsonify(result_data), 200

    except Exception as e:
        logging.error(f"Unexpected error in open format request: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)