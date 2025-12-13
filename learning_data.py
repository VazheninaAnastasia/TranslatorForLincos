# Импорт необходимых библиотек
import json                      # для работы с JSON-файлами
import pandas as pd              # для загрузки и обработки табличных данных
import torch                     # основная библиотека PyTorch
import torch.nn as nn            # модули нейросетей
import torch.optim as optim      # оптимизаторы
from torch.utils.data import Dataset, DataLoader  # для создания пользовательских датасетов и загрузчиков
import matplotlib                # для визуализации
# matplotlib.use('Agg')         # (закомментировано) позволяет рендерить графики без GUI (для серверов)
import matplotlib.pyplot as plt  # построение графиков
import re                        # регулярные выражения для токенизации
from collections import Counter  # подсчёт частоты слов
import os                        # работа с файловой системой


# Класс конфигурации гиперпараметров модели
class Config:
    def __init__(self):
        self.batch_size = 32          # размер мини-батча
        self.embedding_dim = 256      # размерность эмбеддингов
        self.hidden_dim = 512         # размер скрытого состояния LSTM
        self.num_layers = 2           # количество слоёв в LSTM
        self.dropout = 0.3            # вероятность отключения нейронов для регуляризации
        self.learning_rate = 0.001    # скорость обучения
        self.epochs = 40              # количество эпох обучения
        self.max_length = 50          # максимальная длина последовательности (с учётом <SOS> и <EOS>)
        # Автоматический выбор устройства: GPU, если доступен, иначе CPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Класс Vocabulary — управляет словарём токенов (отображение слов в индексы и наоборот)
class Vocabulary:
    def __init__(self):
        # Специальные токены:
        # <PAD> — заполнение до одинаковой длины
        # <SOS> — начало последовательности
        # <EOS> — конец последовательности
        # <UNK> — неизвестное слово
        self.word2idx = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2, '<UNK>': 3}
        self.idx2word = {0: '<PAD>', 1: '<SOS>', 2: '<EOS>', 3: '<UNK>'}
        self.word_count = Counter()   # счётчик частоты слов

    def build_vocabulary(self, sentences, min_freq=1):
        """Создаёт словарь на основе списка предложений."""
        for sentence in sentences:
            words = self.tokenize(sentence)
            self.word_count.update(words)

        # Добавляем слова, встретившиеся не реже min_freq раз
        for word, count in self.word_count.items():
            if count >= min_freq and word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word

    def tokenize(self, text):
        """Разбивает текст на токены: слова, знаки препинания, математические символы."""
        # Регулярное выражение:
        # \b\w+\b — обычные слова,
        # [+\-*/=∈∀∃→∧∨¬<>≤≥] — математические и логические символы,
        # [.,!?;] — пунктуация
        tokens = re.findall(r'\b\w+\b|[+*/=∈∀∃→∧∨¬<>≤≥-]|[.,!?;]', text)
        # Приведение букв к нижнему регистру, но не для символов
        return [token.lower() if token.isalpha() else token for token in tokens]

    def numericalize(self, text):
        """Преобразует текст в последовательность индексов."""
        tokens = self.tokenize(text)
        return [self.word2idx.get(token, self.word2idx['<UNK>']) for token in tokens]

    def __len__(self):
        return len(self.word2idx)  # возвращает размер словаря


# Класс датасета для параллельного корпуса перевода (русский → Lincos)
class TranslationDataset(Dataset):
    def __init__(self, russian_sentences, lincos_sentences, ru_vocab, lincos_vocab, max_length):
        self.russian_sentences = russian_sentences
        self.lincos_sentences = lincos_sentences
        self.ru_vocab = ru_vocab
        self.lincos_vocab = lincos_vocab
        self.max_length = max_length

    def __len__(self):
        return len(self.russian_sentences)

    def __getitem__(self, idx):
        # Получаем пару предложений
        russian_text = self.russian_sentences[idx]
        lincos_text = self.lincos_sentences[idx]

        # Преобразуем в числовые последовательности с <SOS> и <EOS>
        ru_numerical = [self.ru_vocab.word2idx['<SOS>']] + \
                       self.ru_vocab.numericalize(russian_text) + \
                       [self.ru_vocab.word2idx['<EOS>']]

        lincos_numerical = [self.lincos_vocab.word2idx['<SOS>']] + \
                           self.lincos_vocab.numericalize(lincos_text) + \
                           [self.lincos_vocab.word2idx['<EOS>']]

        # Обрезаем до max_length
        ru_numerical = ru_numerical[:self.max_length]
        lincos_numerical = lincos_numerical[:self.max_length]

        # Добавляем паддинг до фиксированной длины
        ru_padding = [self.ru_vocab.word2idx['<PAD>']] * (self.max_length - len(ru_numerical))
        lincos_padding = [self.lincos_vocab.word2idx['<PAD>']] * (self.max_length - len(lincos_numerical))

        # Преобразуем в тензоры
        ru_tensor = torch.tensor(ru_numerical + ru_padding, dtype=torch.long)
        lincos_tensor = torch.tensor(lincos_numerical + lincos_padding, dtype=torch.long)

        return ru_tensor, lincos_tensor


# Энкодер: преобразует входную последовательность в скрытое состояние
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)  # эмбеддинг входных токенов
        self.rnn = nn.LSTM(emb_dim, hidden_dim, num_layers, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        embedded = self.dropout(self.embedding(src))        # применяем эмбеддинг и dropout
        outputs, (hidden, cell) = self.rnn(embedded)        # пропускаем через LSTM
        return hidden, cell                                 # возвращаем скрытые состояния


# Декодер: генерирует целевую последовательность по состоянию от энкодера
class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.output_dim = output_dim
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hidden_dim, num_layers, dropout=dropout, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, output_dim)     # линейный слой для предсказания следующего токена
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell):
        # input — текущий токен (скаляр на батч)
        input = input.unsqueeze(1)                          # добавляем размерность последовательности (batch, 1)
        embedded = self.dropout(self.embedding(input))
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        prediction = self.fc_out(output.squeeze(1))         # (batch, output_dim)
        return prediction, hidden, cell


# Полная Seq2Seq модель: объединяет энкодер и декодер
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        """
        src: (batch, src_len) — входные русские предложения
        trg: (batch, trg_len) — целевые Lincos-предложения
        teacher_forcing_ratio — вероятность использовать ground truth вместо предсказания на шаге t
        """
        batch_size = trg.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.output_dim

        # Массив для хранения выходов на каждом шаге
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)

        # Получаем скрытые состояния от энкодера
        hidden, cell = self.encoder(src)

        # Первый вход декодеру — <SOS>
        input = trg[:, 0]  # (batch,)

        # Генерация токенов пошагово
        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[:, t] = output

            # Решаем: использовать ли true токен (teacher forcing) или предсказанный
            teacher_force = torch.rand(1) < teacher_forcing_ratio
            top1 = output.argmax(1)  # наиболее вероятный токен
            input = trg[:, t] if teacher_force else top1

        return outputs


# === ФУНКЦИИ ЗАГРУЗКИ ДАННЫХ ===

def load_data():
    """Загружает данные из CSV или JSONL файлов."""
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
    """(Не используется в основном коде) — загрузка предварительно сохранённого словаря Lincos."""
    try:
        with open('vocabulary/lincos_vocabulary.json', 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
        return vocab_data
    except FileNotFoundError:
        print("Словарь линкос не найден, будет создан новый")
        return None


# === ФУНКЦИИ ОБУЧЕНИЯ И ОЦЕНКИ ===

def train_model(model, iterator, optimizer, criterion, clip):
    """Обучает модель на одной эпохе."""
    model.train()
    epoch_loss = 0

    for i, (src, trg) in enumerate(iterator):
        src = src.to(config.device)
        trg = trg.to(config.device)

        optimizer.zero_grad()
        output = model(src, trg)  # (batch, trg_len, output_dim)

        # Убираем первый токен (<SOS>) из выхода и целей при подсчёте лосса
        output_dim = output.shape[-1]
        output = output[:, 1:].reshape(-1, output_dim)  # (batch*(trg_len-1), output_dim)
        trg = trg[:, 1:].reshape(-1)                    # (batch*(trg_len-1),)

        loss = criterion(output, trg)
        loss.backward()

        # Обрезка градиентов для предотвращения взрыва градиентов
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def evaluate_model(model, iterator, criterion):
    """Оценивает модель без обучения (без teacher forcing)."""
    model.eval()
    epoch_loss = 0

    with torch.no_grad():
        for i, (src, trg) in enumerate(iterator):
            src = src.to(config.device)
            trg = trg.to(config.device)

            output = model(src, trg, 0)  # teacher_forcing_ratio=0 → всегда используем предсказания

            output_dim = output.shape[-1]
            output = output[:, 1:].reshape(-1, output_dim)
            trg = trg[:, 1:].reshape(-1)

            loss = criterion(output, trg)
            epoch_loss += loss.item()

    return epoch_loss / len(iterator)


# === ОСНОВНАЯ ФУНКЦИЯ ===

def main():
    global config
    config = Config()

    print(f"Используемое устройство: {config.device}")

    # Загрузка данных
    train_df, test_df = load_data()
    if train_df is None:
        return

    print(f"Размер тренировочных данных: {len(train_df)}")
    print(f"Размер тестовых данных: {len(test_df)}")

    # Извлечение текстов из DataFrame
    russian_train = train_df['russian'].tolist()
    lincos_train = train_df['lincos'].tolist()
    russian_test = test_df['russian'].tolist()
    lincos_test = test_df['lincos'].tolist()

    # Построение словарей на основе обучающего и тестового корпуса
    ru_vocab = Vocabulary()
    lincos_vocab = Vocabulary()
    ru_vocab.build_vocabulary(russian_train + russian_test)
    lincos_vocab.build_vocabulary(lincos_train + lincos_test)

    print(f"Размер русского словаря: {len(ru_vocab)}")
    print(f"Размер словаря линкос: {len(lincos_vocab)}")

    # Создание датасетов и загрузчиков
    train_dataset = TranslationDataset(russian_train, lincos_train, ru_vocab, lincos_vocab, config.max_length)
    test_dataset = TranslationDataset(russian_test, lincos_test, ru_vocab, lincos_vocab, config.max_length)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    # Инициализация модели
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

    # Оптимизатор и функция потерь (игнорируем PAD при подсчёте loss)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=lincos_vocab.word2idx['<PAD>'])

    # Списки для отслеживания лосса
    train_losses = []
    test_losses = []
    best_test_loss = float('inf')

    # Цикл обучения
    for epoch in range(config.epochs):
        train_loss = train_model(model, train_loader, optimizer, criterion, clip=1)
        test_loss = evaluate_model(model, test_loader, criterion)

        train_losses.append(train_loss)
        test_losses.append(test_loss)

        print(f'Epoch: {epoch + 1:02}')
        print(f'\tTrain Loss: {train_loss:.3f}')
        print(f'\tTest Loss: {test_loss:.3f}')

        # Каждые 5 эпох выполняется внешний скрипт testing.py (необычное поведение — возможно, для промежуточной проверки)
        if ((epoch + 1) % 5 == 0):
            with open('testing.py', 'r') as file:
                code = file.read()
                exec(code)  # ⚠️ осторожно: выполнение стороннего кода

        # Сохранение лучшей модели по валидационному лоссу
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

    # Построение и сохранение графика лосса
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

    # Сохранение словарей в JSON
    os.makedirs('models', exist_ok=True)
    with open('models/ru_vocab.json', 'w', encoding='utf-8') as f:
        json.dump(ru_vocab.word2idx, f, ensure_ascii=False, indent=2)
    with open('models/lincos_vocab.json', 'w', encoding='utf-8') as f:
        json.dump(lincos_vocab.word2idx, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()