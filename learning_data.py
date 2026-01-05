import json
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib
import matplotlib.pyplot as plt
import re
from collections import Counter
import os


class Config:
    """Конфигурация параметров модели и обучения.

    Атрибуты:
        batch_size (int): Количество примеров в одном батче.
        embedding_dim (int): Размерность слов.
        hidden_dim (int): Размер скрытого состояния в LSTM.
        num_layers (int): Количество слоёв в LSTM.
        dropout (float): Вероятность отключения для регуляризации.
        learning_rate (float): Скорость обучения оптимизатора Adam.
        epochs (int): Общее количество эпох обучения.
        max_length (int): Максимальная длина последовательности (с учётом токенов <SOS>, <EOS>).
        device (torch.device): Устройство для вычислений — GPU, если доступен, иначе CPU.
    """

    def __init__(self):
        self.batch_size = 32
        self.embedding_dim = 256
        self.hidden_dim = 512
        self.num_layers = 2
        self.dropout = 0.3
        self.learning_rate = 0.001
        self.epochs = 40
        self.max_length = 150
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Vocabulary:
    """Управляет словарём токенов: от построения до перевода слов в индексы и обратно.

    Инициализируется с базовыми служебными токенами: <PAD>, <SOS>, <EOS>, <UNK>.
    """

    def __init__(self):
        self.word2idx = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2, '<UNK>': 3}
        self.idx2word = {0: '<PAD>', 1: '<SOS>', 2: '<EOS>', 3: '<UNK>'}
        self.word_count = Counter()

    def build_vocabulary(self, sentences, min_freq=1):
        """Строит словарь на основе списка предложений.

        Токенизирует все предложения, считает частоту токенов и добавляет в словарь
        только те, чья частота >= min_freq.

        Args:
            sentences (list[str]): Список текстовых предложений.
            min_freq (int): Минимальная частота токена для включения в словарь.
        """
        for sentence in sentences:
            words = self.tokenize(sentence)
            self.word_count.update(words)

        for word, count in self.word_count.items():
            if count >= min_freq and word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word

    def tokenize(self, text):
        """Разбивает текст на токены: слова, математические символы, знаки препинания.

        Латинские и кириллические слова приводятся к нижнему регистру.
        Специальные символы (например, ∈, →, ∀) сохраняются как есть.

        Args:
            text (str): Исходный текст.

        Returns:
            list[str]: Список токенов.
        """
        tokens = re.findall(r'\b\w+\b|[+*/=∈∀∃→∧∨¬<>≤≥-]|[.,!?;]', text)
        return [token.lower() if token.isalpha() else token for token in tokens]

    def numericalize(self, text):
        """Преобразует текст в последовательность индексов из словаря.

        Неизвестные токены заменяются на <UNK>.

        Args:
            text (str): Исходный текст.

        Returns:
            list[int]: Список индексов.
        """
        tokens = self.tokenize(text)
        return [self.word2idx.get(token, self.word2idx['<UNK>']) for token in tokens]

    def __len__(self):
        """Возвращает размер словаря (количество уникальных токенов)."""
        return len(self.word2idx)


class TranslationDataset(Dataset):
    """Набор данных для задачи перевода с русского на Lincos (или наоборот).

    Каждый элемент — пара тензоров фиксированной длины (с паддингом),
    представляющих исходное и целевое предложения, обрамлённые токенами <SOS>/<EOS>.
    """

    def __init__(self, russian_sentences, lincos_sentences, ru_vocab, lincos_vocab, max_length):
        """
        Args:
            russian_sentences (list[str]): Список предложений на русском.
            lincos_sentences (list[str]): Список соответствующих предложений на Lincos.
            ru_vocab (Vocabulary): Словарь для русского языка.
            lincos_vocab (Vocabulary): Словарь для Lincos.
            max_length (int): Максимальная длина последовательности (включая <SOS>/<EOS>).
        """
        self.russian_sentences = russian_sentences
        self.lincos_sentences = lincos_sentences
        self.ru_vocab = ru_vocab
        self.lincos_vocab = lincos_vocab
        self.max_length = max_length

    def __len__(self):
        """Возвращает общее количество пар предложений."""
        return len(self.russian_sentences)

    def __getitem__(self, idx):
        """Возвращает пару тензоров: (русское предложение, Lincos-предложение).

        Оба тензора:
          - начинаются с <SOS>, заканчиваются <EOS>,
          - обрезаются до max_length,
          - дополняются <PAD> до фиксированной длины.

        Args:
            idx (int): Индекс пары.

        Returns:
            tuple[torch.LongTensor, torch.LongTensor]: Тензоры длины max_length.
        """
        russian_text = self.russian_sentences[idx]
        lincos_text = self.lincos_sentences[idx]

        ru_numerical = [self.ru_vocab.word2idx['<SOS>']] + \
                       self.ru_vocab.numericalize(russian_text) + \
                       [self.ru_vocab.word2idx['<EOS>']]

        lincos_numerical = [self.lincos_vocab.word2idx['<SOS>']] + \
                           self.lincos_vocab.numericalize(lincos_text) + \
                           [self.lincos_vocab.word2idx['<EOS>']]

        ru_numerical = ru_numerical[:self.max_length]
        lincos_numerical = lincos_numerical[:self.max_length]

        ru_padding = [self.ru_vocab.word2idx['<PAD>']] * (self.max_length - len(ru_numerical))
        lincos_padding = [self.lincos_vocab.word2idx['<PAD>']] * (self.max_length - len(lincos_numerical))

        ru_tensor = torch.tensor(ru_numerical + ru_padding, dtype=torch.long)
        lincos_tensor = torch.tensor(lincos_numerical + lincos_padding, dtype=torch.long)

        return ru_tensor, lincos_tensor


class Encoder(nn.Module):
    """Кодировщик последовательности на основе LSTM.

    Преобразует входную последовательность (например, на русском) в скрытое состояние,
    которое передаётся декодеру.
    """

    def __init__(self, input_dim, emb_dim, hidden_dim, num_layers, dropout):
        """
        Args:
            input_dim (int): Размер входного словаря.
            emb_dim (int): Размерность эмбеддингов.
            hidden_dim (int): Размерность скрытого состояния LSTM.
            num_layers (int): Количество слоёв LSTM.
            dropout (float): Вероятность dropout после эмбеддинга и между LSTM-слоями.
        """
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hidden_dim, num_layers, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        """Пропускает последовательность через кодировщик.

        Args:
            src (torch.LongTensor): Входной тензор формы (batch_size, seq_len).

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Кортеж (hidden, cell) последнего слоя LSTM.
        """
        embedded = self.dropout(self.embedding(src))
        outputs, (hidden, cell) = self.rnn(embedded)
        return hidden, cell


class Decoder(nn.Module):
    """Декодер последовательности на основе LSTM.

    Генерирует выходную последовательность (например, на Lincos) пошагово,
    используя скрытое состояние от кодировщика и предыдущие предсказания.
    """

    def __init__(self, output_dim, emb_dim, hidden_dim, num_layers, dropout):
        """
        Args:
            output_dim (int): Размер выходного словаря.
            emb_dim (int): Размерность эмбеддингов.
            hidden_dim (int): Размерность скрытого состояния LSTM.
            num_layers (int): Количество слоёв LSTM.
            dropout (float): Вероятность dropout после эмбеддинга.
        """
        super().__init__()
        self.output_dim = output_dim
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hidden_dim, num_layers, dropout=dropout, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell):
        """Генерирует следующий токен на основе текущего входа и состояния LSTM.

        Args:
            input (torch.LongTensor): Текущий входной токен, форма (batch_size,).
            hidden (torch.Tensor): Скрытое состояние от предыдущего шага.
            cell (torch.Tensor): Ячейка памяти LSTM от предыдущего шага.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - логиты для следующего токена (batch_size, output_dim),
                - новое скрытое состояние,
                - новая ячейка памяти.
        """
        input = input.unsqueeze(1)  # (batch_size, 1)
        embedded = self.dropout(self.embedding(input))
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        prediction = self.fc_out(output.squeeze(1))
        return prediction, hidden, cell


class Seq2Seq(nn.Module):
    """Полная модель sequence-to-sequence с кодировщиком и декодером.

    Поддерживает teacher forcing во время обучения.
    """

    def __init__(self, encoder, decoder, device):
        """
        Args:
            encoder (Encoder): Экземпляр кодировщика.
            decoder (Decoder): Экземпляр декодера.
            device (torch.device): Устройство для тензоров.
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        """Выполняет прямой проход модели.

        Args:
            src (torch.LongTensor): Входная последовательность, форма (batch_size, src_len).
            trg (torch.LongTensor): Целевая последовательность, форма (batch_size, trg_len).
            teacher_forcing_ratio (float): Вероятность использовать ground-truth токен вместо предсказанного.

        Returns:
            torch.Tensor: Выходные логиты, форма (batch_size, trg_len, output_dim).
        """
        batch_size = trg.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.output_dim

        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
        hidden, cell = self.encoder(src)
        input = trg[:, 0]  # <SOS> токен

        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[:, t] = output
            teacher_force = torch.rand(1) < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = trg[:, t] if teacher_force else top1

        return outputs


def load_data():
    """Загружает обучающие и тестовые данные из CSV или JSONL.

    Сначала пытается найти файлы в формате CSV:
        - lincos_dataset/train/train.csv
        - lincos_dataset/test/test.csv
    Если не найдены — ищет JSONL-файлы:
        - train/train.jsonl
        - test/test.jsonl

    Возвращает pandas.DataFrame с колонками ['russian', 'lincos'].

    Returns:
        tuple[pd.DataFrame, pd.DataFrame] | tuple[None, None]:
            Обучающий и тестовый датафреймы, либо (None, None), если файлы не найдены.
    """
    try:
        train_df = pd.read_csv('lincos_dataset/train/train.csv')
        test_df = pd.read_csv('lincos_dataset/test/test.csv')
        return train_df, test_df
    except FileNotFoundError:
        print("CSV файлы не найдены, пробуем загрузить из JSONL...")
        try:
            train_data = []
            with open('train/train.jsonl', 'r', encoding='utf-8') as f:
                for line in f:
                    train_data.append(json.loads(line))

            test_data = []
            with open('test/test.jsonl', 'r', encoding='utf-8') as f:
                for line in f:
                    test_data.append(json.loads(line))

            train_df = pd.DataFrame(train_data)
            test_df = pd.DataFrame(test_data)
            return train_df, test_df
        except FileNotFoundError:
            print("Файлы данных не найдены!")
            return None, None


def load_vocabulary():
    """Загружает предварительно сохранённый словарь Lincos из JSON-файла (устаревшая функция).

    В текущей реализации словарь строится динамически, но функция оставлена на случай
    восстановления из кэша.

    Returns:
        dict | None: Словарь word → index или None, если файл не найден.
    """
    try:
        with open('vocabulary/lincos_vocabulary.json', 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
        return vocab_data
    except FileNotFoundError:
        print("Словарь линкос не найден, будет создан новый")
        return None


def train_model(model, iterator, optimizer, criterion, clip):
    """Выполняет одну эпоху обучения модели.

    Args:
        model (Seq2Seq): Обучаемая модель.
        iterator (DataLoader): Итератор по обучающим данным.
        optimizer (torch.optim.Optimizer): Оптимизатор.
        criterion (nn.Module): Функция потерь.
        clip (float): Максимальная норма градиента для clipping'а.

    Returns:
        float: Среднее значение функции потерь за эпоху.
    """
    model.train()
    epoch_loss = 0

    for i, (src, trg) in enumerate(iterator):
        src = src.to(config.device)
        trg = trg.to(config.device)

        optimizer.zero_grad()
        output = model(src, trg)

        output_dim = output.shape[-1]
        output = output[:, 1:].reshape(-1, output_dim)
        trg = trg[:, 1:].reshape(-1)

        loss = criterion(output, trg)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def evaluate_model(model, iterator, criterion):
    """Выполняет оценку модели на валидационном/тестовом наборе (без teacher forcing).

    Args:
        model (Seq2Seq): Модель для оценки.
        iterator (DataLoader): Итератор по данным.
        criterion (nn.Module): Функция потерь.

    Returns:
        float: Среднее значение функции потерь.
    """
    model.eval()
    epoch_loss = 0

    with torch.no_grad():
        for i, (src, trg) in enumerate(iterator):
            src = src.to(config.device)
            trg = trg.to(config.device)

            output = model(src, trg, 0)  # teacher_forcing_ratio = 0

            output_dim = output.shape[-1]
            output = output[:, 1:].reshape(-1, output_dim)
            trg = trg[:, 1:].reshape(-1)

            loss = criterion(output, trg)
            epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def main():
    """Основная функция обучения: загрузка данных, построение словарей, обучение и сохранение модели."""
    global config
    config = Config()

    print(f"Используемое устройство: {config.device}")

    train_df, test_df = load_data()
    if train_df is None:
        return

    print(f"Размер тренировочных данных: {len(train_df)}")
    print(f"Размер тестовых данных: {len(test_df)}")

    russian_train = train_df['russian'].tolist()
    lincos_train = train_df['lincos'].tolist()
    russian_test = test_df['russian'].tolist()
    lincos_test = test_df['lincos'].tolist()

    ru_vocab = Vocabulary()
    lincos_vocab = Vocabulary()
    ru_vocab.build_vocabulary(russian_train + russian_test)
    lincos_vocab.build_vocabulary(lincos_train + lincos_test)

    print(f"Размер русского словаря: {len(ru_vocab)}")
    print(f"Размер словаря линкос: {len(lincos_vocab)}")

    train_dataset = TranslationDataset(russian_train, lincos_train, ru_vocab, lincos_vocab, config.max_length)
    test_dataset = TranslationDataset(russian_test, lincos_test, ru_vocab, lincos_vocab, config.max_length)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    encoder = Encoder(
        input_dim=len(ru_vocab),
        emb_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        dropout=config.dropout
    )
    decoder = Decoder(
        output_dim=len(lincos_vocab),
        emb_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        dropout=config.dropout
    )
    model = Seq2Seq(encoder, decoder, config.device).to(config.device)

    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=lincos_vocab.word2idx['<PAD>'])

    train_losses = []
    test_losses = []
    best_test_loss = float('inf')

    for epoch in range(config.epochs):
        train_loss = train_model(model, train_loader, optimizer, criterion, clip=1)
        test_loss = evaluate_model(model, test_loader, criterion)

        train_losses.append(train_loss)
        test_losses.append(test_loss)

        print(f'Epoch: {epoch + 1:02}')
        print(f'\tTrain Loss: {train_loss:.3f}')
        print(f'\tTest Loss: {test_loss:.3f}')

        if ((epoch + 1) % 5 == 0):
            with open('testing.py', 'r') as file:
                code = file.read()
                exec(code)

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'test_loss': test_loss,
                'ru_vocab': ru_vocab,
                'lincos_vocab': lincos_vocab
            }, 'best_model.pth')

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.savefig('training_loss.png')
    print("График сохранен как 'training_loss.png'")

    print("Обучение завершено!")

    os.makedirs('models', exist_ok=True)
    with open('models/ru_vocab.json', 'w', encoding='utf-8') as f:
        json.dump(ru_vocab.word2idx, f, ensure_ascii=False, indent=2)
    with open('models/lincos_vocab.json', 'w', encoding='utf-8') as f:
        json.dump(lincos_vocab.word2idx, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()