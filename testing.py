import torch
from learning_data import Vocabulary, Encoder, Decoder, Seq2Seq, Config


class EnhancedTranslator:
    """Инференс-обёртка для перевода с русского языка на Lincos.

    Загружает предобученную модель и словари из checkpoint'а и предоставляет
    удобный метод translate() для перевода отдельных предложений.
    Работает в режиме оценки (model.eval()), без градиентов.
    """

    def __init__(self, model_path='best_model.pth'):
        """Инициализирует переводчик, загружая модель и словари из файла.

        Args:
            model_path (str): Путь к файлу с сохранённой моделью (.pth).
                              Должен содержать: веса модели, ru_vocab, lincos_vocab.
        """
        self.config = Config()
        # weights_only=False — требуется для совместимости с PyTorch < 2.4
        checkpoint = torch.load(model_path, map_location=self.config.device, weights_only=False)

        self.ru_vocab = checkpoint['ru_vocab']
        self.lincos_vocab = checkpoint['lincos_vocab']

        # Создаём архитектуру модели (без dropout'а — он не нужен при инференсе)
        encoder = Encoder(
            input_dim=len(self.ru_vocab),
            emb_dim=self.config.embedding_dim,
            hidden_dim=self.config.hidden_dim,
            num_layers=self.config.num_layers,
            dropout=0.0
        )

        decoder = Decoder(
            output_dim=len(self.lincos_vocab),
            emb_dim=self.config.embedding_dim,
            hidden_dim=self.config.hidden_dim,
            num_layers=self.config.num_layers,
            dropout=0.0
        )

        self.model = Seq2Seq(encoder, decoder, self.config.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()  # Важно: отключаем dropout и batch norm для детерминированного вывода

    def translate(self, russian_text):
        """Переводит одно предложение с русского на Lincos.

        Текст токенизируется, преобразуется в последовательность индексов,
        дополняется до максимальной длины, подаётся в кодировщик,
        а затем декодер генерирует выход пошагово (без teacher forcing).

        Args:
            russian_text (str): Предложение на русском языке.

        Returns:
            str: Перевод на Lincos в виде строки токенов, разделённых пробелами.
                 Служебные токены (<SOS>, <EOS>, <PAD>) удаляются из результата.
        """
        # Токенизация и числовое представление с обрамлением <SOS>/<EOS>
        tokens = self.ru_vocab.tokenize(russian_text)
        numerical = [self.ru_vocab.word2idx['<SOS>']] + \
                    [self.ru_vocab.word2idx.get(token, self.ru_vocab.word2idx['<UNK>']) for token in tokens] + \
                    [self.ru_vocab.word2idx['<EOS>']]

        # Приведение к фиксированной длине (хотя при инференсе это не всегда обязательно)
        if len(numerical) < self.config.max_length:
            numerical += [self.ru_vocab.word2idx['<PAD>']] * (self.config.max_length - len(numerical))
        else:
            numerical = numerical[:self.config.max_length]

        src_tensor = torch.tensor(numerical).unsqueeze(0).to(self.config.device)  # (1, max_len)

        # Кодировка входного предложения
        with torch.no_grad():
            hidden, cell = self.model.encoder(src_tensor)

        # Декодирование: начинаем с <SOS>
        trg_indices = [self.lincos_vocab.word2idx['<SOS>']]

        # Генерация токенов до <EOS> или достижения лимита
        for _ in range(self.config.max_length - 1):
            trg_tensor = torch.tensor([trg_indices[-1]]).to(self.config.device)
            with torch.no_grad():
                output, hidden, cell = self.model.decoder(trg_tensor, hidden, cell)
            pred_token = output.argmax(1).item()
            trg_indices.append(pred_token)
            if pred_token == self.lincos_vocab.word2idx['<EOS>']:
                break

        # Преобразование индексов в слова, исключая служебные токены
        lincos_words = []
        for idx in trg_indices[1:]:  # пропускаем <SOS>
            if idx == self.lincos_vocab.word2idx['<EOS>']:
                break
            word = self.lincos_vocab.idx2word.get(idx, '<UNK>')
            # Не включаем служебные токены в финальный результат
            if word not in ['<PAD>', '<SOS>', '<EOS>']:
                lincos_words.append(word)

        return ' '.join(lincos_words)


if __name__ == "__main__":
    """Демонстрация работы переводчика на нескольких примерах."""
    translator = EnhancedTranslator()

    test_cases = [
        "два плюс два равно четыре",
        "пять минус три равно два",
        "y является элементом множества Com",
        "если x равно c то c равно y",
        "шесть умножить на восемь равно cорок восемь",
        "двенадцать делить на шесть равно два",
        "восемь делить на четыре равно два"
    ]

    for text in test_cases:
        result = translator.translate(text)
        print(f"Русский: {text}")
        print(f"Lincos:  {result}")
        print()