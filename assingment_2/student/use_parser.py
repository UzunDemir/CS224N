#!/usr/bin/env python3
import torch
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.parser_utils import load_and_preprocess_data
from parser_model import ParserModel

def load_trained_model(model_path="results/20251117_155032/model.weights"):
    """Загружает обученную модель с полным датасетом"""
    print("Loading model with full dataset...")

    # Загружаем ПОЛНЫЙ датасет
    parser, embeddings, _, _, _ = load_and_preprocess_data(reduced=False)

    model = ParserModel(embeddings)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    parser.model = model
    parser.model.eval()

    return parser

def create_conll_sentence(sentence_words):
    """Создает предложение в формате CONLL"""
    # Создаем минимальную CONLL структуру
    conll_sentence = {
        'word': sentence_words,
        'pos': ['NN'] * len(sentence_words),  # заглушка для POS тегов
        'head': [0] * len(sentence_words),    # заглушка для голов
        'label': ['dep'] * len(sentence_words) # заглушка для меток
    }
    return conll_sentence

def parse_sentences(parser, sentences):
    """Парсит список предложений"""
    all_dependencies = []

    for sentence in sentences:
        print(f"\nParsing: {' '.join(sentence)}")
        print("-" * 50)

        try:
            # Создаем CONLL предложение
            conll_sentence = create_conll_sentence(sentence)

            # Векторизуем предложение
            vectorized_sentences = parser.vectorize([conll_sentence])

            # Парсим и получаем (UAS, dependencies)
            UAS, dependencies_list = parser.parse(vectorized_sentences)

            # dependencies_list содержит зависимости для всех предложений
            dependencies = dependencies_list[0]  # берем первое предложение

            print(f"UAS: {UAS:.2%}")
            print("\nDependencies:")
            print("-" * 30)

            for head_idx, dep_idx in dependencies:
                if head_idx == 0:  # ROOT
                    head_word = "ROOT"
                else:
                    head_word = sentence[head_idx-1] if head_idx-1 < len(sentence) else f"word_{head_idx}"

                if dep_idx == 0:  # ROOT (маловероятно, но на всякий случай)
                    dep_word = "ROOT"
                else:
                    dep_word = sentence[dep_idx-1] if dep_idx-1 < len(sentence) else f"word_{dep_idx}"

                print(f"{head_word:>12} → {dep_word}")

        except Exception as e:
            print(f"Error parsing sentence: {e}")
            import traceback
            traceback.print_exc()

    return all_dependencies

def interactive_mode(parser):
    """Интерактивный режим"""
    print("\n" + "="*50)
    print("Interactive Dependency Parser")
    print("Type sentences to parse (or 'quit' to exit)")
    print("="*50)

    while True:
        try:
            user_input = input("\nEnter sentence: ").strip()
            if user_input.lower() in ['quit', 'exit', 'q']:
                break

            if not user_input:
                continue

            sentence = user_input.split()
            if len(sentence) < 2:
                print("Sentence too short! Need at least 2 words.")
                continue

            parse_sentences(parser, [sentence])

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")

def test_with_simple_sentences(parser):
    """Тестирует с простыми предложениями"""
    print("Testing with simple sentences...")

    simple_sentences = [
        ["the", "cat", "sat"],
        ["i", "like", "dogs"],
        ["she", "reads", "books"],
        ["the", "quick", "brown", "fox", "jumps"]
    ]

    parse_sentences(parser, simple_sentences)

def comprehensive_test(parser):
    """Комплексный тест разных типов предложений"""
    print("\n" + "="*60)
    print("COMPREHENSIVE PARSER TEST")
    print("="*60)

    test_sentences = [
        # Простые предложения
        ["the", "cat", "sleeps"],
        ["i", "love", "python"],
        ["she", "reads", "books"],

        # Предложения с предлогами
        ["the", "cat", "sat", "on", "the", "mat"],
        ["he", "went", "to", "the", "store"],

        # Предложения с прилагательными
        ["the", "big", "black", "cat", "sleeps"],
        ["she", "has", "beautiful", "blue", "eyes"],

        # Вопросительные предложения
        ["does", "the", "cat", "sleep"],
        ["what", "do", "you", "want"],

        # Сложные предложения
        ["the", "cat", "that", "sat", "on", "the", "mat", "was", "black"]
    ]

    for i, sentence in enumerate(test_sentences, 1):
        print(f"\n{'='*40}")
        print(f"TEST {i}: {' '.join(sentence)}")
        print('='*40)

        try:
            conll_sentence = create_conll_sentence(sentence)
            vectorized = parser.vectorize([conll_sentence])
            UAS, dependencies_list = parser.parse(vectorized)
            dependencies = dependencies_list[0]

            print(f"UAS: {UAS:.2%}")
            print("\nDependencies:")
            print("-" * 25)

            # Сортируем для лучшей читаемости (по зависимому слову)
            sorted_deps = sorted(dependencies, key=lambda x: x[1])

            for head_idx, dep_idx in sorted_deps:
                if head_idx == 0:
                    head_word = "ROOT"
                else:
                    head_word = sentence[head_idx-1] if head_idx-1 < len(sentence) else f"#{head_idx}"

                if dep_idx == 0:
                    dep_word = "ROOT"
                else:
                    dep_word = sentence[dep_idx-1] if dep_idx-1 < len(sentence) else f"#{dep_idx}"

                print(f"{head_word:>12} → {dep_word}")

        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    try:
        parser = load_trained_model()
        print("✓ Model loaded successfully!")

        # Тестируем с простыми предложениями
        test_with_simple_sentences(parser)

        # Комплексный тест
        comprehensive_test(parser)

        # Интерактивный режим
        interactive_mode(parser)

    except Exception as e:
        print(f"Error: {e}")