# Скрипт для загрузки обученной модели и перевода русских фраз на искусственный язык Lincos

import torch
# Импортируем необходимые компоненты модели и вспомогательные классы из основного модуля обучения
from learning_data import Vocabulary, Encoder, Decoder, Seq2Seq, Config


class EnhancedTranslator:
    """Класс для перевода русского текста на язык Lincos с использованием обученной Seq2Seq-модели."""

    def __init__(self, model_path='best_model.pth'):
        # Загружаем конфигурацию (гиперпараметры, устройство и т.д.)
        self.config = Config()

        # Загружаем сохранённую модель и словари из файла
        # `weights_only=False` необходимо при загрузке пользовательских объектов (например, Vocabulary)
        checkpoint = torch.load(model_path, map_location=self.config.device, weights_only=False)

        # Восстанавливаем словари из чекпоинта
        self.ru_vocab = checkpoint['ru_vocab']      # словарь русского языка
        self.lincos_vocab = checkpoint['lincos_vocab']  # словарь Lincos

        # Инициализируем архитектуру модели (точно такую же, как при обучении)
        # Dropout отключён (0.0), так как мы находимся в режиме инференса (оценки)
        encoder = Encoder(
            input_dim=len(self.ru_vocab),           # размер входного словаря (русский)
            emb_dim=self.config.embedding_dim,
            hidden_dim=self.config.hidden_dim,
            num_layers=self.config.num_layers,
            dropout=0.0                             # без dropout при инференсе
        )

        decoder = Decoder(
            output_dim=len(self.lincos_vocab),      # размер выходного словаря (Lincos)
            emb_dim=self.config.embedding_dim,
            hidden_dim=self.config.hidden_dim,
            num_layers=self.config.num_layers,
            dropout=0.0
        )

        # Создаём и загружаем обученные веса в Seq2Seq-модель
        self.model = Seq2Seq(encoder, decoder, self.config.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()  # Включаем evaluation mode (отключает dropout, batch norm и т.п.)

    def translate(self, russian_text):
        """
        Переводит одну фразу с русского на Lincos.
        :param russian_text: строка на русском языке
        :return: строка на языке Lincos
        """
        # Токенизация входного текста с помощью того же токенизатора, что использовался при обучении
        tokens = self.ru_vocab.tokenize(russian_text)

        # Преобразуем токены в числовую последовательность, добавляя специальные токены <SOS> и <EOS>
        numerical = [self.ru_vocab.word2idx['<SOS>']] + \
                    [self.ru_vocab.word2idx.get(token, self.ru_vocab.word2idx['<UNK>']) for token in tokens] + \
                    [self.ru_vocab.word2idx['<EOS>']]

        # Приводим последовательность к фиксированной длине (максимум config.max_length)
        if len(numerical) < self.config.max_length:
            # Добавляем паддинг, если текст короче максимальной длины
            numerical += [self.ru_vocab.word2idx['<PAD>']] * (self.config.max_length - len(numerical))
        else:
            # Обрезаем, если текст слишком длинный
            numerical = numerical[:self.config.max_length]

        # Преобразуем в тензор и добавляем размерность батча (batch_size = 1)
        src_tensor = torch.tensor(numerical).unsqueeze(0).to(self.config.device)

        # Пропускаем вход через энкодер, получая начальные скрытые состояния для декодера
        with torch.no_grad():  # отключаем вычисление градиентов — мы не обучаемся
            hidden, cell = self.model.encoder(src_tensor)

        # Инициализируем целевую последовательность с токеном начала <SOS>
        trg_indices = [self.lincos_vocab.word2idx['<SOS>']]

        # Постепенно генерируем токены до достижения максимальной длины или токена <EOS>
        for _ in range(self.config.max_length - 1):
            # Берём последний предсказанный токен как вход для следующего шага
            trg_tensor = torch.tensor([trg_indices[-1]]).to(self.config.device)
            with torch.no_grad():
                output, hidden, cell = self.model.decoder(trg_tensor, hidden, cell)
            # Выбираем токен с наибольшей вероятностью
            pred_token = output.argmax(1).item()
            trg_indices.append(pred_token)
            # Прерываем генерацию, если модель выдала конец последовательности
            if pred_token == self.lincos_vocab.word2idx['<EOS>']:
                break

        # Преобразуем полученные индексы обратно в слова, исключая служебные токены
        lincos_words = []
        for idx in trg_indices[1:]:  # пропускаем <SOS>
            if idx == self.lincos_vocab.word2idx['<EOS>']:
                break
            word = self.lincos_vocab.idx2word.get(idx, '<UNK>')
            # Исключаем служебные токены из финального вывода
            if word not in ['<PAD>', '<SOS>', '<EOS>']:
                lincos_words.append(word)

        # Возвращаем результат в виде строки
        return ' '.join(lincos_words)


# Точка входа: демонстрация работы переводчика на тестовых примерах
if __name__ == "__main__":
    # Инициализируем переводчик (загружает модель из 'best_model.pth')
    translator = EnhancedTranslator()

    # Список тестовых фраз на русском языке (включая арифметику и логические выражения)
    test_cases = [
        "два плюс два равно четыре",
        "пять минус три равно два",
        "y является элементом множества Com",
        "если x равно c то c равно y",
        "шесть умножить на восемь равно cорок восемь",
        "двенадцать делить на шесть равно два",
        "восемь делить на четыре равно два"
    ]

    # Переводим каждую фразу и выводим пару «русский → Lincos»
    for text in test_cases:
        result = translator.translate(text)
        print(f"Русский: {text}")
        print(f"Lincos:  {result}")
        print()  # пустая строка для читаемости