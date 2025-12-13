# visual_for_user.py
# Графический интерфейс пользователя (GUI) для переводчика русского текста на искусственный язык Lincos.
# Используется библиотека Flet (современный фреймворк для создания кроссплатформенных приложений на Python).

import flet as ft              # Основная библиотека для GUI
import torch                   # Для загрузки и работы с PyTorch-моделью
# Импортируем компоненты модели и конфигурации из основного модуля обучения
from learning_data import Encoder, Decoder, Seq2Seq, Config


class TranslatorApp:
    """Класс-обёртка для загрузки модели и выполнения перевода текста."""

    def __init__(self):
        self.config = Config()      # Загружаем конфигурацию (гиперпараметры, устройство)
        self.translator = None      # Зарезервировано (не используется напрямую)
        self.load_model()           # Загружаем модель при инициализации

    def load_model(self):
        """Загружает обученную модель и словари из файла 'best_model.pth'."""
        try:
            # Загружаем чекпоинт, перемещая его на текущее устройство (CPU/GPU)
            checkpoint = torch.load('best_model.pth', map_location=self.config.device, weights_only=False)

            # Восстанавливаем словари из сохранённого состояния
            self.ru_vocab = checkpoint['ru_vocab']
            self.lincos_vocab = checkpoint['lincos_vocab']

            # Пересоздаём архитектуру модели (точно такую же, как при обучении)
            encoder = Encoder(
                input_dim=len(self.ru_vocab),
                emb_dim=self.config.embedding_dim,
                hidden_dim=self.config.hidden_dim,
                num_layers=self.config.num_layers,
                dropout=0.0  # Dropout отключён — режим инференса
            )

            decoder = Decoder(
                output_dim=len(self.lincos_vocab),
                emb_dim=self.config.embedding_dim,
                hidden_dim=self.config.hidden_dim,
                num_layers=self.config.num_layers,
                dropout=0.0
            )

            # Создаём Seq2Seq-модель и загружаем обученные веса
            self.model = Seq2Seq(encoder, decoder, self.config.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()  # Включаем evaluation mode

            return True
        except Exception as e:
            print(f"Ошибка загрузки модели: {e}")
            return False

    def translate_text(self, russian_text):
        """
        Выполняет перевод одной фразы с русского на Lincos.
        Возвращает строку с результатом или сообщение об ошибке.
        """
        if not russian_text.strip():
            return ""

        try:
            # Токенизация и преобразование в числовую последовательность (аналогично обучению)
            tokens = self.ru_vocab.tokenize(russian_text)
            numerical = [self.ru_vocab.word2idx['<SOS>']] + \
                        [self.ru_vocab.word2idx.get(token, self.ru_vocab.word2idx['<UNK>']) for token in tokens] + \
                        [self.ru_vocab.word2idx['<EOS>']]

            # Приведение к фиксированной длине
            if len(numerical) < self.config.max_length:
                numerical += [self.ru_vocab.word2idx['<PAD>']] * (self.config.max_length - len(numerical))
            else:
                numerical = numerical[:self.config.max_length]

            # Преобразуем в тензор и добавляем размерность батча
            src_tensor = torch.tensor(numerical).unsqueeze(0).to(self.config.device)

            # Пропускаем через энкодер
            with torch.no_grad():
                hidden, cell = self.model.encoder(src_tensor)

            # Инициализируем декодирование с <SOS>
            trg_indices = [self.lincos_vocab.word2idx['<SOS>']]

            # Авто-регрессивная генерация токенов
            for _ in range(self.config.max_length - 1):
                trg_tensor = torch.tensor([trg_indices[-1]]).to(self.config.device)
                with torch.no_grad():
                    output, hidden, cell = self.model.decoder(trg_tensor, hidden, cell)
                pred_token = output.argmax(1).item()
                trg_indices.append(pred_token)
                if pred_token == self.lincos_vocab.word2idx['<EOS>']:
                    break

            # Преобразуем индексы в слова, исключая служебные токены
            lincos_words = []
            for idx in trg_indices[1:]:
                if idx == self.lincos_vocab.word2idx['<EOS>']:
                    break
                word = self.lincos_vocab.idx2word.get(idx, '<UNK>')
                if word not in ['<PAD>', '<SOS>', '<EOS>']:
                    lincos_words.append(word)

            return ' '.join(lincos_words)

        except Exception as e:
            return f"Ошибка перевода: {e}"


def main(page: ft.Page):
    """
    Основная функция Flet-приложения: настраивает страницу и строит интерфейс.
    :param page: объект страницы Flet, управляющий содержимым окна
    """

    # === НАСТРОЙКИ СТРАНИЦЫ ===
    page.title = "Транслятор с русского на Lincos"
    page.theme_mode = ft.ThemeMode.LIGHT  # Светлая тема
    page.padding = 20                     # Внешний отступ
    page.window.width = 900               # Ширина окна
    page.window.height = 800              # Высота окна
    page.window.resizable = True          # Разрешить изменение размера

    # === ИНИЦИАЛИЗАЦИЯ ПЕРЕВОДЧИКА ===
    translator_app = TranslatorApp()

    # === ЗАГОЛОВОК ===
    title = ft.Container(
        content=ft.Column([
            ft.Row([
                ft.Text("Транслятор с русского на Lincos",
                        size=28,
                        weight=ft.FontWeight.BOLD,
                        color=ft.Colors.BLUE_900),
            ], alignment=ft.MainAxisAlignment.CENTER),
            ft.Divider(height=10, color=ft.Colors.TRANSPARENT),
        ]),
        margin=ft.margin.only(bottom=30)
    )

    # === ПОЛЕ ВВОДА ===
    input_field = ft.TextField(
        label="Введите текст на русском",
        multiline=True,            # Многострочный ввод
        min_lines=3,
        max_lines=5,
        border_color=ft.Colors.BLUE_400,
        focused_border_color=ft.Colors.BLUE_700,
        hint_text="Например: два плюс два равно четыре",
        expand=True,
    )

    # === КНОПКИ ===
    # ИСПРАВЛЕНИЕ 1: ft.Icons вместо ft.icons
    translate_button = ft.ElevatedButton(
        content=ft.Row([
            ft.Icon(ft.Icons.TRANSLATE, color=ft.Colors.WHITE),
            ft.Text("Перевести", color=ft.Colors.WHITE, weight=ft.FontWeight.BOLD),
        ]),
        bgcolor=ft.Colors.BLUE_600,
        color=ft.Colors.WHITE,
        on_click=lambda _: translate_click(),  # Обработчик нажатия
        style=ft.ButtonStyle(
            shape=ft.RoundedRectangleBorder(radius=10),
            padding=20,
        ),
        height=50,
    )

    delete_button = ft.ElevatedButton(
        content=ft.Row([
            ft.Icon(ft.Icons.DELETE, color=ft.Colors.WHITE),
            ft.Text("Сбросить", color=ft.Colors.WHITE, weight=ft.FontWeight.BOLD),
        ]),
        bgcolor=ft.Colors.GREY_600,
        color=ft.Colors.WHITE,
        on_click=lambda _: delete_click(),
        style=ft.ButtonStyle(
            shape=ft.RoundedRectangleBorder(radius=10),
            padding=20,
        ),
        height=50,
    )

    # === ПОЛЕ ВЫВОДА РЕЗУЛЬТАТА ===
    output_text = ft.Text(
        "",
        size=18,
        weight=ft.FontWeight.W_500,
        color=ft.Colors.GREEN_800
    )

    output_field = ft.Container(
        content=ft.Column([
            ft.Text("Результат перевода на линкос:",
                    size=16,
                    weight=ft.FontWeight.BOLD,
                    color=ft.Colors.BLUE_900),
            ft.Container(
                content=output_text,
                padding=15,
                border_radius=10,
                bgcolor=ft.Colors.GREEN_50,           # Светло-зелёный фон
                border=ft.border.all(2, ft.Colors.GREEN_200),
            )
        ]),
        visible=False,  # Скрыто до первого перевода
    )

    # === ПРИМЕРЫ ДЛЯ БЫСТРОГО ТЕСТИРОВАНИЯ ===
    examples_title = ft.Text("Примеры для быстрого тестирования:",
                             size=16,
                             weight=ft.FontWeight.BOLD,
                             color=ft.Colors.GREY_800)

    examples = [
        "два плюс два равно четыре",
        "пять минус три равно два",
        "шесть умножить на шесть равно тридцать шесть",
        "y является элементом множества Com",
        "если y равно c то c равно y",
        "для всех z если z принадлежит Rat то z принадлежит Rea"
    ]

    example_buttons = ft.Row(
        wrap=True,          # Перенос кнопок на новую строку при нехватке места
        spacing=10,         # Горизонтальный отступ между кнопками
        run_spacing=10,     # Вертикальный отступ между строками кнопок
    )

    # Создаём кнопки для каждого примера
    for example in examples:
        # Используем `text=example` в лямбде, чтобы захватить текущее значение
        example_btn = ft.TextButton(
            content=ft.Container(
                content=ft.Text(example, size=12, color=ft.Colors.BLUE_600),
                padding=10,
            ),
            style=ft.ButtonStyle(
                shape=ft.RoundedRectangleBorder(radius=8),
                side=ft.BorderSide(1, ft.Colors.BLUE_300),
                bgcolor=ft.Colors.BLUE_50,
            ),
            on_click=lambda e, text=example: example_click(text),
        )
        example_buttons.controls.append(example_btn)

    # === ИНДИКАТОР ЗАГРУЗКИ ===
    progress_ring = ft.ProgressRing(visible=False)  # Круговая анимация загрузки

    # === СТАТУС ЗАГРУЗКИ МОДЕЛИ ===
    # ИСПРАВЛЕНИЕ 2: ft.Icons вместо ft.icons
    model_status = ft.Container(
        content=ft.Row([
            ft.Icon(ft.Icons.CHECK_CIRCLE, color=ft.Colors.GREEN, size=20),
            ft.Text("Модель загружена и готова к работе",
                    color=ft.Colors.GREEN_700),
        ]),
        bgcolor=ft.Colors.GREEN_50,
        padding=10,
        border_radius=8,
        margin=ft.margin.only(bottom=20),
    )

    # === ФУНКЦИИ-ОБРАБОТЧИКИ ===

    def translate_click():
        """Выполняет перевод при нажатии на кнопку 'Перевести'."""
        if not input_field.value.strip():
            return

        # Показываем индикатор загрузки и блокируем кнопку
        progress_ring.visible = True
        translate_button.disabled = True
        page.update()

        # Выполняем перевод
        result = translator_app.translate_text(input_field.value)

        # Обновляем результат
        output_text.value = result
        output_field.visible = True

        # Скрываем индикатор и разблокируем кнопку
        progress_ring.visible = False
        translate_button.disabled = False
        page.update()

    def delete_click():
        """Очищает поле ввода и скрывает результат."""
        input_field.value = ""
        output_field.visible = False
        output_text.value = ""
        input_field.focus()  # Возвращаем фокус в поле ввода
        page.update()

    def example_click(text):
        """Подставляет пример в поле ввода и запускает перевод."""
        input_field.value = text
        page.update()
        translate_click()

    # === СБОРКА ИНТЕРФЕЙСА ===
    page.add(
        title,
        model_status,
        ft.Container(
            content=ft.Column([
                # Карточка с вводом
                ft.Card(
                    content=ft.Container(
                        content=ft.Column([
                            ft.Text("Ввод текста",
                                    size=18,
                                    weight=ft.FontWeight.BOLD,
                                    color=ft.Colors.BLUE_900),
                            input_field,
                            ft.Row([
                                translate_button,
                                delete_button,
                                progress_ring,  # Индикатор рядом с кнопками
                            ], alignment=ft.MainAxisAlignment.START),
                        ], spacing=15),
                        padding=20,
                    ),
                    elevation=5,
                    margin=ft.margin.only(bottom=20),
                ),

                # Поле вывода (изначально скрыто)
                output_field,

                # Карточка с примерами
                ft.Card(
                    content=ft.Container(
                        content=ft.Column([
                            examples_title,
                            example_buttons,
                        ], spacing=15),
                        padding=20,
                    ),
                    elevation=3,
                ),
            ], spacing=20),
        )
    )


# === ЗАПУСК ПРИЛОЖЕНИЯ ===
if __name__ == "__main__":
    ft.app(target=main)  # Запускает Flet-приложение