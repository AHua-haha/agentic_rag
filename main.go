package main

import (
	"bufio"
	"fmt"
	"llm_dev/agent"
	_ "llm_dev/utils"
	"os"
)

var sss string

func main() {

	agent.InitVectorDB()
	defer agent.CloseVectorDB()

	model := agent.NewModel("http://172.17.0.1:4000", "sk-1234")
	agent := agent.RagAgent{
		BaseAgent: agent.NewBaseAgent("", *model),
	}
	agent.SetSysprompt()
	for {
		reader := bufio.NewScanner(os.Stdin)
		fmt.Print("User Prompt> ")
		reader.Scan() // This will read a line of input from the user
		userprompt := reader.Text()

		agent.NewUserTask(userprompt)
	}
}
