"""
Script to calculate quiz accuracy statistics across different dimensions:
1. Average quiz accuracy for different products
2. Quiz accuracy based on difficulty levels
3. Average accuracy for each product at different complexity levels
4. Average accuracy for all products at each complexity level
"""

import json
import csv
import argparse
from collections import defaultdict
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(
        description='Calculate quiz accuracy statistics from quantitative evaluation results'
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to quantitative results JSON file'
    )
    parser.add_argument(
        '--quiz-data',
        type=str,
        required=True,
        help='Path to quiz data JSON file (e.g., final_revise.json)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='results',
        help='Output directory for CSV files (default: results)'
    )
    return parser.parse_args()

args = parse_args()

# Load the data files
print("Loading data files...")
with open(args.input, 'r') as f:
    quant_results = json.load(f)

with open(args.quiz_data, 'r') as f:
    final_revise = json.load(f)

# Create a mapping from filename to document info in final_revise
print("Creating filename mappings...")
filename_to_doc = {}
for doc in final_revise['documents']:
    filename = doc['meta_info']['filename']
    # Remove extension and create variations
    base_filename = filename.replace('.pdf', '')
    filename_to_doc[base_filename] = doc
    filename_to_doc[filename] = doc

# Also create a mapping by topic/ppt_id
topic_to_doc = {}
for doc in final_revise['documents']:
    # Try to match by source path components
    source_path = doc['meta_info']['source_path']
    # Extract the folder name (topic)
    parts = source_path.split('/')
    if len(parts) >= 3:
        topic = parts[-2]  # The folder name before the filename
        topic_to_doc[topic] = doc

print(f"Mapped {len(filename_to_doc)} filenames and {len(topic_to_doc)} topics")

# Group "lite" + "Remaining" as "topic introduction"
def normalize_difficulty(difficulty):
    """Normalize difficulty levels, grouping lite and Remaining as topic introduction"""
    if difficulty.lower() in ['lite', 'remaining']:
        return 'topic introduction'
    return difficulty.lower()

# Process quiz results
print("\nProcessing quiz results...")
results_by_product = defaultdict(list)
results_by_difficulty = defaultdict(list)
results_by_product_complexity = defaultdict(lambda: defaultdict(list))
results_by_product_difficulty = defaultdict(lambda: defaultdict(list))
results_by_complexity = defaultdict(list)
detailed_results = []
# Track questions by topic to analyze failure rates
questions_by_topic = defaultdict(lambda: {'products': defaultdict(list), 'doc': None})

for result in quant_results['results']:
    # Skip entries where quiz_details is null (no quiz data available)
    if result.get('quiz_details') is None:
        continue
    
    product = result['product']
    difficulty = normalize_difficulty(result['difficulty'])
    topic = result['topic']
    quiz_accuracy = result.get('quiz_accuracy', 0)
    
    # Try to find matching document in final_revise
    doc = None
    if topic in topic_to_doc:
        doc = topic_to_doc[topic]
    elif topic in filename_to_doc:
        doc = filename_to_doc[topic]
    else:
        # Try partial match
        for key, value in topic_to_doc.items():
            if topic in key or key in topic:
                doc = value
                break
    
    # Get complexity level and domain from doc if found
    complexity_level = doc['statistics']['complexity_level'] if doc else 'Unknown'
    domain = doc['meta_info']['domain'] if doc else 'Unknown'
    
    # Store results
    results_by_product[product].append(quiz_accuracy)
    results_by_difficulty[difficulty].append(quiz_accuracy)
    results_by_product_complexity[product][complexity_level].append(quiz_accuracy)
    results_by_product_difficulty[product][difficulty].append(quiz_accuracy)
    results_by_complexity[complexity_level].append(quiz_accuracy)
    
    # Track individual quiz answers for each topic
    questions_by_topic[topic]['products'][product].append(quiz_accuracy)
    if questions_by_topic[topic]['doc'] is None:
        questions_by_topic[topic]['doc'] = doc
    
    # Store detailed result
    detailed_results.append({
        'product': product,
        'difficulty': difficulty,
        'topic': topic,
        'domain': domain,
        'complexity_level': complexity_level,
        'quiz_accuracy': quiz_accuracy
    })

print(f"Processed {len(detailed_results)} results")

# Analyze questions where all products failed
print("\n=== Analyzing questions with 0% accuracy across all products ===")
failed_questions_by_difficulty = defaultdict(lambda: {'total': 0, 'all_failed': 0})
failed_questions_by_complexity = defaultdict(lambda: {'total': 0, 'all_failed': 0})
failed_questions_by_domain = defaultdict(lambda: {'total': 0, 'all_failed': 0})
all_failed_topics = []
# Store individual questions that all products failed
all_failed_questions_detailed = []

for topic, data in questions_by_topic.items():
    doc = data['doc']
    products_data = data['products']
    
    # Check if all products got 0% accuracy for this topic
    all_accuracies = [acc for accs in products_data.values() for acc in accs]
    max_accuracy = max(all_accuracies) if all_accuracies else 0
    
    # Get topic metadata
    if doc:
        complexity = doc['statistics']['complexity_level']
        domain = doc['meta_info']['domain']
        # Find difficulty from detailed_results
        difficulty = None
        for result in detailed_results:
            if result['topic'] == topic:
                difficulty = result['difficulty']
                break
        if difficulty is None:
            difficulty = 'Unknown'
    else:
        complexity = 'Unknown'
        domain = 'Unknown'
        difficulty = 'Unknown'
    
    # Count total topics and failed topics by dimension
    failed_questions_by_difficulty[difficulty]['total'] += 1
    failed_questions_by_complexity[complexity]['total'] += 1
    failed_questions_by_domain[domain]['total'] += 1
    
    if max_accuracy == 0:
        # All products failed on this topic
        failed_questions_by_difficulty[difficulty]['all_failed'] += 1
        failed_questions_by_complexity[complexity]['all_failed'] += 1
        failed_questions_by_domain[domain]['all_failed'] += 1
        
        all_failed_topics.append({
            'topic': topic,
            'difficulty': difficulty,
            'complexity': complexity,
            'domain': domain,
            'num_products_tested': len(products_data)
        })
        
        # If we have the document, extract individual questions that failed
        if doc and 'quiz_data' in doc and 'quiz_bank' in doc['quiz_data']:
            for question in doc['quiz_data']['quiz_bank']:
                all_failed_questions_detailed.append({
                    'topic': topic,
                    'difficulty': difficulty,
                    'complexity': complexity,
                    'domain': domain,
                    'question_id': question['id'],
                    'question_type': question.get('type', 'Unknown'),
                    'question_text': question.get('question', ''),
                    'correct_answer': question.get('correct_answer', ''),
                    'explanation': question.get('explanation', ''),
                    'filename': doc['meta_info']['filename'],
                    'num_products_tested': len(products_data)
                })

print(f"Found {len(all_failed_topics)} topics where all products scored 0%")

# Calculate failure ratios
def calculate_failure_ratio(stats_dict):
    return {
        category: {
            'total_topics': stats['total'],
            'all_failed_topics': stats['all_failed'],
            'failure_ratio': stats['all_failed'] / stats['total'] if stats['total'] > 0 else 0
        }
        for category, stats in stats_dict.items()
    }

failure_by_difficulty = calculate_failure_ratio(failed_questions_by_difficulty)
failure_by_complexity = calculate_failure_ratio(failed_questions_by_complexity)
failure_by_domain = calculate_failure_ratio(failed_questions_by_domain)

# Calculate averages
def calculate_average(values):
    """Calculate average, handling empty lists"""
    return sum(values) / len(values) if values else 0

# 1. Average quiz accuracy by product
print("\n=== Calculating average accuracy by product ===")
product_averages = {
    product: calculate_average(accuracies)
    for product, accuracies in results_by_product.items()
}

# 2. Average quiz accuracy by difficulty
print("=== Calculating average accuracy by difficulty ===")
difficulty_averages = {
    difficulty: calculate_average(accuracies)
    for difficulty, accuracies in results_by_difficulty.items()
}

# 3. Average accuracy for each product at different complexity levels
print("=== Calculating average accuracy by product and complexity ===")
product_complexity_averages = {}
for product, complexity_dict in results_by_product_complexity.items():
    product_complexity_averages[product] = {
        complexity: calculate_average(accuracies)
        for complexity, accuracies in complexity_dict.items()
    }

# 3b. Average accuracy for each product at different difficulty levels
print("=== Calculating average accuracy by product and difficulty ===")
product_difficulty_averages = {}
for product, difficulty_dict in results_by_product_difficulty.items():
    product_difficulty_averages[product] = {
        difficulty: calculate_average(accuracies)
        for difficulty, accuracies in difficulty_dict.items()
    }

# 4. Average accuracy for all products at each complexity level
print("=== Calculating average accuracy by complexity level ===")
complexity_averages = {
    complexity: calculate_average(accuracies)
    for complexity, accuracies in results_by_complexity.items()
}

# Output results to CSV files
output_dir = Path(args.output)
output_dir.mkdir(parents=True, exist_ok=True)

print("\n=== Writing CSV files ===")

# CSV 1: Average accuracy by product
csv1_path = output_dir / 'quiz_accuracy_by_product.csv'
with open(csv1_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Product', 'Average Quiz Accuracy', 'Number of Samples'])
    for product in sorted(product_averages.keys()):
        writer.writerow([
            product,
            f"{product_averages[product]:.4f}",
            len(results_by_product[product])
        ])
print(f"✓ Written: {csv1_path}")

# CSV 2: Average accuracy by difficulty
csv2_path = output_dir / 'quiz_accuracy_by_difficulty.csv'
with open(csv2_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Difficulty', 'Average Quiz Accuracy', 'Number of Samples'])
    for difficulty in sorted(difficulty_averages.keys()):
        writer.writerow([
            difficulty,
            f"{difficulty_averages[difficulty]:.4f}",
            len(results_by_difficulty[difficulty])
        ])
print(f"✓ Written: {csv2_path}")

# CSV 3: Average accuracy by complexity level (all products)
csv3_path = output_dir / 'quiz_accuracy_by_complexity.csv'
with open(csv3_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Complexity Level', 'Average Quiz Accuracy', 'Number of Samples'])
    for complexity in sorted(complexity_averages.keys()):
        writer.writerow([
            complexity,
            f"{complexity_averages[complexity]:.4f}",
            len(results_by_complexity[complexity])
        ])
print(f"✓ Written: {csv3_path}")

# CSV 4: Average accuracy by product and complexity level
csv4_path = output_dir / 'quiz_accuracy_by_product_and_complexity.csv'
all_complexities = sorted(set(complexity for product_dict in product_complexity_averages.values() 
                               for complexity in product_dict.keys()))
with open(csv4_path, 'w', newline='') as f:
    writer = csv.writer(f)
    header = ['Product'] + all_complexities + ['Overall Average']
    writer.writerow(header)
    
    for product in sorted(product_complexity_averages.keys()):
        row = [product]
        for complexity in all_complexities:
            if complexity in product_complexity_averages[product]:
                row.append(f"{product_complexity_averages[product][complexity]:.4f}")
            else:
                row.append('N/A')
        row.append(f"{product_averages[product]:.4f}")
        writer.writerow(row)
    
    # Add overall averages row
    row = ['Overall Average']
    for complexity in all_complexities:
        row.append(f"{complexity_averages[complexity]:.4f}")
    row.append(f"{calculate_average([acc for accs in results_by_product.values() for acc in accs]):.4f}")
    writer.writerow(row)
print(f"✓ Written: {csv4_path}")

# CSV 4b: Average accuracy by product and difficulty level
csv4b_path = output_dir / 'quiz_accuracy_by_product_and_difficulty.csv'
all_difficulties = sorted(set(difficulty for product_dict in product_difficulty_averages.values() 
                               for difficulty in product_dict.keys()))
with open(csv4b_path, 'w', newline='') as f:
    writer = csv.writer(f)
    header = ['Product'] + all_difficulties + ['Overall Average']
    writer.writerow(header)
    
    for product in sorted(product_difficulty_averages.keys()):
        row = [product]
        for difficulty in all_difficulties:
            if difficulty in product_difficulty_averages[product]:
                row.append(f"{product_difficulty_averages[product][difficulty]:.4f}")
            else:
                row.append('N/A')
        row.append(f"{product_averages[product]:.4f}")
        writer.writerow(row)
    
    # Add overall averages row
    row = ['Overall Average']
    for difficulty in all_difficulties:
        row.append(f"{difficulty_averages[difficulty]:.4f}")
    row.append(f"{calculate_average([acc for accs in results_by_product.values() for acc in accs]):.4f}")
    writer.writerow(row)
print(f"✓ Written: {csv4b_path}")

# CSV 5: Detailed results with all fields
csv5_path = output_dir / 'quiz_accuracy_detailed.csv'
with open(csv5_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Product', 'Difficulty', 'Topic', 'Domain', 'Complexity Level', 'Quiz Accuracy'])
    for result in detailed_results:
        writer.writerow([
            result['product'],
            result['difficulty'],
            result['topic'],
            result['domain'],
            result['complexity_level'],
            f"{result['quiz_accuracy']:.4f}"
        ])
print(f"✓ Written: {csv5_path}")

# CSV 6: All-failed topics analysis by difficulty
csv6_path = output_dir / 'all_failed_questions_by_difficulty.csv'
with open(csv6_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Difficulty', 'Total Topics', 'All Failed Topics', 'Failure Ratio', 'Failure Percentage'])
    for difficulty in sorted(failure_by_difficulty.keys()):
        stats = failure_by_difficulty[difficulty]
        writer.writerow([
            difficulty,
            stats['total_topics'],
            stats['all_failed_topics'],
            f"{stats['failure_ratio']:.4f}",
            f"{stats['failure_ratio']*100:.2f}%"
        ])
print(f"✓ Written: {csv6_path}")

# CSV 7: All-failed topics analysis by complexity
csv7_path = output_dir / 'all_failed_questions_by_complexity.csv'
with open(csv7_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Complexity Level', 'Total Topics', 'All Failed Topics', 'Failure Ratio', 'Failure Percentage'])
    for complexity in sorted(failure_by_complexity.keys()):
        stats = failure_by_complexity[complexity]
        writer.writerow([
            complexity,
            stats['total_topics'],
            stats['all_failed_topics'],
            f"{stats['failure_ratio']:.4f}",
            f"{stats['failure_ratio']*100:.2f}%"
        ])
print(f"✓ Written: {csv7_path}")

# CSV 8: All-failed topics analysis by domain
csv8_path = output_dir / 'all_failed_questions_by_domain.csv'
with open(csv8_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Domain', 'Total Topics', 'All Failed Topics', 'Failure Ratio', 'Failure Percentage'])
    for domain in sorted(failure_by_domain.keys()):
        stats = failure_by_domain[domain]
        writer.writerow([
            domain,
            stats['total_topics'],
            stats['all_failed_topics'],
            f"{stats['failure_ratio']:.4f}",
            f"{stats['failure_ratio']*100:.2f}%"
        ])
print(f"✓ Written: {csv8_path}")

# CSV 9: List of all topics that failed across all products
csv9_path = output_dir / 'all_failed_topics_list.csv'
with open(csv9_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Topic', 'Difficulty', 'Complexity Level', 'Domain', 'Number of Products Tested'])
    for failed_topic in sorted(all_failed_topics, key=lambda x: (x['difficulty'], x['complexity'], x['topic'])):
        writer.writerow([
            failed_topic['topic'],
            failed_topic['difficulty'],
            failed_topic['complexity'],
            failed_topic['domain'],
            failed_topic['num_products_tested']
        ])
print(f"✓ Written: {csv9_path}")

# CSV 10: Detailed list of individual questions that all products failed
csv10_path = output_dir / 'failed_questions_detailed.csv'
with open(csv10_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['Topic', 'Filename', 'Difficulty', 'Complexity Level', 'Domain', 
                     'Question ID', 'Question Type', 'Question Text', 'Correct Answer', 
                     'Explanation', 'Number of Products Tested'])
    for question in sorted(all_failed_questions_detailed, 
                          key=lambda x: (x['difficulty'], x['complexity'], x['topic'], x['question_id'])):
        writer.writerow([
            question['topic'],
            question['filename'],
            question['difficulty'],
            question['complexity'],
            question['domain'],
            question['question_id'],
            question['question_type'],
            question['question_text'],
            question['correct_answer'],
            question['explanation'],
            question['num_products_tested']
        ])
print(f"✓ Written: {csv10_path}")
print(f"  Found {len(all_failed_questions_detailed)} individual questions where all products failed")

# Extract failed questions from final_revise.json into a new JSON file
print("\n=== Extracting failed questions to separate JSON file ===")
failed_questions_json = {
    'generated_at': final_revise.get('generated_at', ''),
    'summary': {
        'total_failed_topics': len(all_failed_topics),
        'total_failed_questions': len(all_failed_questions_detailed),
        'total_topics_in_dataset': len(questions_by_topic),
        'failure_rate': len(all_failed_topics) / len(questions_by_topic) if len(questions_by_topic) > 0 else 0
    },
    'failed_documents': []
}

# Get unique topics that failed
failed_topics_set = {topic['topic'] for topic in all_failed_topics}

# Extract full documents for failed topics
for doc in final_revise['documents']:
    filename = doc['meta_info']['filename']
    base_filename = filename.replace('.pdf', '')
    source_path = doc['meta_info']['source_path']
    
    # Check if this document corresponds to a failed topic
    is_failed = False
    for failed_topic in failed_topics_set:
        if (failed_topic == base_filename or 
            failed_topic in source_path or 
            base_filename in failed_topic):
            is_failed = True
            break
    
    if is_failed:
        failed_questions_json['failed_documents'].append(doc)

json_output_path = output_dir / 'failed_questions_extracted.json'
with open(json_output_path, 'w', encoding='utf-8') as f:
    json.dump(failed_questions_json, f, indent=2, ensure_ascii=False)
print(f"✓ Written: {json_output_path}")
print(f"  Extracted {len(failed_questions_json['failed_documents'])} failed documents")

# Print summary statistics
print("\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)

print(f"\nTotal evaluations: {len(detailed_results)}")
print(f"Number of products: {len(product_averages)}")
print(f"Number of difficulty levels: {len(difficulty_averages)}")
print(f"Number of complexity levels: {len(complexity_averages)}")

print("\n--- Top 5 Products by Quiz Accuracy ---")
sorted_products = sorted(product_averages.items(), key=lambda x: x[1], reverse=True)
for i, (product, avg) in enumerate(sorted_products[:5], 1):
    print(f"{i}. {product}: {avg:.4f} ({len(results_by_product[product])} samples)")

print("\n--- Quiz Accuracy by Difficulty Level ---")
for difficulty in sorted(difficulty_averages.keys()):
    avg = difficulty_averages[difficulty]
    count = len(results_by_difficulty[difficulty])
    print(f"{difficulty}: {avg:.4f} ({count} samples)")

print("\n--- Quiz Accuracy by Complexity Level ---")
for complexity in sorted(complexity_averages.keys()):
    avg = complexity_averages[complexity]
    count = len(results_by_complexity[complexity])
    print(f"{complexity}: {avg:.4f} ({count} samples)")

print("\n" + "="*80)
print("QUESTIONS WHERE ALL PRODUCTS FAILED (0% Accuracy)")
print("="*80)

print(f"\nTotal topics where all products scored 0%: {len(all_failed_topics)}")
print(f"Total individual questions that all products failed: {len(all_failed_questions_detailed)}")
print(f"Percentage of all topics: {len(all_failed_topics)/len(questions_by_topic)*100:.2f}%")

print("\n--- Failure Ratio by Difficulty Level ---")
for difficulty in sorted(failure_by_difficulty.keys()):
    stats = failure_by_difficulty[difficulty]
    print(f"{difficulty}: {stats['all_failed_topics']}/{stats['total_topics']} ({stats['failure_ratio']*100:.2f}%)")

print("\n--- Failure Ratio by Complexity Level ---")
for complexity in sorted(failure_by_complexity.keys()):
    stats = failure_by_complexity[complexity]
    print(f"{complexity}: {stats['all_failed_topics']}/{stats['total_topics']} ({stats['failure_ratio']*100:.2f}%)")

print("\n--- Failure Ratio by Domain ---")
for domain in sorted(failure_by_domain.keys()):
    stats = failure_by_domain[domain]
    print(f"{domain}: {stats['all_failed_topics']}/{stats['total_topics']} ({stats['failure_ratio']*100:.2f}%)")

print("\n" + "="*80)
print("All CSV files have been generated successfully!")
print("="*80)
