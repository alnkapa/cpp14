#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <algorithm>

using namespace std;

// Функция для чтения тестовых данных
void readTestData(const string& filename, vector<int>& labels, vector<vector<double>>& images) {
    ifstream file(filename);
    string line;

    while (getline(file, line)) {
        stringstream ss(line);
        int label;
        ss >> label;
        labels.push_back(label);

        vector<double> image(784);
        for (int i = 0; i < 784; ++i) {
            ss.ignore(1); // Игнорируем запятую
            ss >> image[i];
        }
        images.push_back(image);
    }
}

// Функция для загрузки коэффициентов логистической регрессии
void loadLogisticRegressionModel(const string& filename, vector<vector<double>>& coefficients) {
    ifstream file(filename);
    double value;
    for (int i = 0; i < 10; ++i) {
        vector<double> row;
        for (int j = 0; j < 785; ++j) {
            file >> value;
            row.push_back(value);
        }
        coefficients.push_back(row);
    }
}

// Функция для предсказания класса
int predict(const vector<double>& image, const vector<vector<double>>& coefficients) {
    vector<double> extendedImage(785);
    extendedImage[0] = 1.0; // bias
    for (int i = 0; i < 784; ++i) {
        extendedImage[i + 1] = image[i];
    }

    // Вычисляем все скалярные произведения
    vector<double> scores(10, 0.0);
    for (int i = 0; i < 10; ++i) {
        for (int j = 0; j < 785; ++j) {
            scores[i] += coefficients[i][j] * extendedImage[j];
        }
    }

    // Находим класс с максимальным значением
    return max_element(scores.begin(), scores.end()) - scores.begin();
}

// Основная функция
int main(int argc, char* argv[]) {
    if (argc != 3) {
        cerr << "Usage: " << argv[0] << " <test.csv> <model>" << endl;
        return 1;
    }

    string testFile = argv[1];
    string modelFile = argv[2];

    vector<int> labels;
    vector<vector<double>> images;
    readTestData(testFile, labels, images);

    vector<vector<double>> coefficients;
    loadLogisticRegressionModel(modelFile, coefficients);

    int correctPredictions = 0;

    for (size_t i = 0; i < images.size(); ++i) {
        int predictedClass = predict(images[i], coefficients);
        if (predictedClass == labels[i]) {
            correctPredictions++;
        }
    }

    double accuracy = static_cast<double>(correctPredictions) / labels.size();
    cout << accuracy << endl;

    return 0;
}
