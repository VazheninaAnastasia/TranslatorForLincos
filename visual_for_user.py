# Графический интерфейс пользователя (GUI) для переводчика русского текста на искусственный язык Lincos.
# Используется библиотека Flet (современный фреймворк для создания кроссплатформенных приложений на Python).

import flet as ft              # Основная библиотека для GUI
import torch                   # Для загрузки и работы с PyTorch-моделью
# Импортируем компоненты модели и конфигурации из основного модуля обучения
from learning_data import Encoder, Decoder, Seq2Seq, Config, Vocabulary


class TranslatorApp:
    """Класс-обёртка для загрузки модели и выполнения перевода текста."""

    def __init__(self):
        self.config = Config()
        self.model = None
        self.ru_vocab = None
        self.lincos_vocab = None
        self.load_model()

    def load_model(self):
        """Загружает обученную модель и словари из файла 'best_model.pth'."""
        try:
            checkpoint = torch.load('best_model.pth', map_location=self.config.device, weights_only=False)

            self.ru_vocab = checkpoint['ru_vocab']
            self.lincos_vocab = checkpoint['lincos_vocab']

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
            self.model.eval()
            return True
        except Exception as e:
            print(f"Ошибка загрузки модели: {e}")
            return False

    def translate_text(self, russian_text):
        """Выполняет перевод одной фразы с русского на Lincos."""
        if not russian_text.strip():
            return ""

        try:
            tokens = self.ru_vocab.tokenize(russian_text)
            numerical = [self.ru_vocab.word2idx['<SOS>']] + \
                        [self.ru_vocab.word2idx.get(token, self.ru_vocab.word2idx['<UNK>']) for token in tokens] + \
                        [self.ru_vocab.word2idx['<EOS>']]

            if len(numerical) < self.config.max_length:
                numerical += [self.ru_vocab.word2idx['<PAD>']] * (self.config.max_length - len(numerical))
            else:
                numerical = numerical[:self.config.max_length]

            src_tensor = torch.tensor(numerical).unsqueeze(0).to(self.config.device)

            with torch.no_grad():
                hidden, cell = self.model.encoder(src_tensor)

            trg_indices = [self.lincos_vocab.word2idx['<SOS>']]
            for _ in range(self.config.max_length - 1):
                trg_tensor = torch.tensor([trg_indices[-1]]).to(self.config.device)
                with torch.no_grad():
                    output, hidden, cell = self.model.decoder(trg_tensor, hidden, cell)
                pred_token = output.argmax(1).item()
                trg_indices.append(pred_token)
                if pred_token == self.lincos_vocab.word2idx['<EOS>']:
                    break

            lincos_words = []
            for idx in trg_indices[1:]:
                if idx == self.lincos_vocab.word2idx['<EOS>']:
                    break
                word = self.lincos_vocab.idx2word.get(idx, '<UNK>')
                if word not in {'<PAD>', '<SOS>', '<EOS>'}:
                    lincos_words.append(word)

            return ' '.join(lincos_words)

        except Exception as e:
            return f"Ошибка перевода: {e}"


def main(page: ft.Page):
    page.title = "Транслятор с русского на Lincos"
    page.theme_mode = ft.ThemeMode.LIGHT
    page.padding = 20
    page.window.width = 900
    page.window.height = 800
    page.window.resizable = True

    translator_app = TranslatorApp()

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

    input_field = ft.TextField(
        label="Введите текст на русском (ограничение на ввод 50 символов)",
        multiline=True,
        min_lines=3,
        max_lines=5,
        border_color=ft.Colors.BLUE_400,
        focused_border_color=ft.Colors.BLUE_700,
        hint_text="Например: два плюс два равно четыре",
        expand=True,
    )

    translate_button = ft.ElevatedButton(
        content=ft.Row([
            ft.Icon(ft.Icons.TRANSLATE, color=ft.Colors.WHITE),
            ft.Text("Перевести", color=ft.Colors.WHITE, weight=ft.FontWeight.BOLD),
        ]),
        bgcolor=ft.Colors.BLUE_600,
        color=ft.Colors.WHITE,
        on_click=lambda _: translate_click(),
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
                bgcolor=ft.Colors.GREEN_50,
                border=ft.border.all(2, ft.Colors.GREEN_200),
            )
        ]),
        visible=False,
    )

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
        wrap=True,
        spacing=10,
        run_spacing=10,
    )

    for example in examples:
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

    progress_ring = ft.ProgressRing(visible=False)

    # Определяем статус модели
    if translator_app.model is None:
        model_status_content = ft.Row([
            ft.Icon(ft.Icons.ERROR, color=ft.Colors.RED, size=20),
            ft.Text("Ошибка загрузки модели. Проверьте наличие best_model.pth.",
                    color=ft.Colors.RED_700),
        ])
        status_bg = ft.Colors.RED_50
    else:
        model_status_content = ft.Row([
            ft.Icon(ft.Icons.CHECK_CIRCLE, color=ft.Colors.GREEN, size=20),
            ft.Text("Модель загружена и готова к работе",
                    color=ft.Colors.GREEN_700),
        ])
        status_bg = ft.Colors.GREEN_50

    model_status = ft.Container(
        content=model_status_content,
        bgcolor=status_bg,
        padding=10,
        border_radius=8,
        margin=ft.margin.only(bottom=20),
    )

    def translate_click():
        if not input_field.value.strip():
            return

        progress_ring.visible = True
        translate_button.disabled = True
        page.update()

        result = translator_app.translate_text(input_field.value)

        output_text.value = result
        output_field.visible = True

        progress_ring.visible = False
        translate_button.disabled = False
        page.update()

    def delete_click():
        input_field.value = ""
        output_field.visible = False
        output_text.value = ""
        input_field.focus()
        page.update()

    def example_click(text):
        input_field.value = text
        page.update()
        translate_click()

    page.add(
        title,
        model_status,
        ft.Container(
            content=ft.Column([
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
                                progress_ring,
                            ], alignment=ft.MainAxisAlignment.START),
                        ], spacing=15),
                        padding=20,
                    ),
                    elevation=5,
                    margin=ft.margin.only(bottom=20),
                ),
                output_field,
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


if __name__ == "__main__":
    ft.app(target=main)