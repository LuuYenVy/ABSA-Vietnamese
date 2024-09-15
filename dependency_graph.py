from py_vncorenlp import VnCoreNLP
import numpy as np
import pickle
import argparse

# Đường dẫn tới mô hình VnCoreNLP
model_path = "D:\Thuc tap\ABSA-PyTorch\models\VnCoreNLP-1.1.1"

# Khởi tạo VnCoreNLP
vnlp = VnCoreNLP(model_path)

def dependency_adj_matrix(text):
    # Phân tích văn bản
    parsed = vnlp.parse(text)
    words = [word for sentence in parsed for word in sentence]
    word_indices = {word[0]: i for i, word in enumerate(words)}
    
    matrix = np.zeros((len(words), len(words))).astype('float32')
    
    for sentence in parsed:
        for word in sentence:
            head = word[1]
            if head != 0:  # 0 có nghĩa là không có head
                matrix[word_indices[head]][word_indices[word[0]]] = 1
                matrix[word_indices[word[0]]][word_indices[head]] = 1
    
    return matrix

def process(filename):
    with open(filename, 'r', encoding='utf-8', newline='\n', errors='ignore') as fin:
        lines = fin.readlines()
    
    idx2graph = {}
    with open(filename + '.graph', 'wb') as fout:
        for i in range(0, len(lines), 3):
            text_left, _, text_right = [s.strip() for s in lines[i].partition("$T$")]
            aspect = lines[i + 1].strip()
            adj_matrix = dependency_adj_matrix(text_left + ' ' + aspect + ' ' + text_right)
            idx2graph[i] = adj_matrix
        pickle.dump(idx2graph, fout)        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default=None, type=str, help='path to dataset')
    opt = parser.parse_args()
    process(opt.dataset)
