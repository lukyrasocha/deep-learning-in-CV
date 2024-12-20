from rich.console import Console

class Logger:
    def __init__(self):
        self.console = Console()

    def working_on(self, message):
        self.console.print(":wrench: [bold green]WORKING ON[/bold green]: " + message)


    def warning(self,message):
        self.console.print(":tomato: [bold red]WARNING[/bold red]: " + message)

    def error(self,message):
        self.console.print(":tomato: [bold red]ERROR[/bold red]: " + message)

    def info(self,message):
        self.console.print(
            ":information_source: [bold blue]INFO[/bold blue]: " + message)

    def success(self,message):
        self.console.print(
            ":white_check_mark: [bold green]SUCCESS[/bold green]: " + message)

    def winner(self,message):
        self.console.print(
            ":trophy: [bold yellow]WINNER[/bold yellow]: " + message)

logger = Logger()

if __name__ == "__main__":
    logger.warning("This is a warning message")
    logger.working_on("This is a working on message")
    logger.info("This is an info message")
    logger.success("This is a success message")