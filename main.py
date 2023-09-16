import os
import logging
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from PyPDF2 import PdfReader
from langchain.vectorstores import ElasticVectorSearch, Pinecone, Weaviate, FAISS

logging.basicConfig(
  format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
  level=logging.INFO)

DATABASE = None


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
  await context.bot.send_message(chat_id=update.effective_chat.id,
                                 text="I'm a bot, please talk to me!")


async def load(update: Update, context: ContextTypes.DEFAULT_TYPE):
  loader = PdfReader('Final_merged new.pdf')
  #documents=loader.load()
  raw_text = ''
  for i, page in enumerate(loader.pages):
    text = page.extract_text()
    if text:
      raw_text += text
  text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
  )
  texts = text_splitter.split_text(raw_text)
  #docs = text_splitter.split_documents(documents)

  global DATABASE
  DATABASE = FAISS.from_texts(texts, OpenAIEmbeddings())
  await context.bot.send_message(chat_id=update.effective_chat.id,
                                 text="Document loaded!")


async def query(update: Update, context: ContextTypes.DEFAULT_TYPE):
  docs = DATABASE.similarity_search(update.message.text, k=4)
  chain = load_qa_chain(llm=OpenAI(), chain_type="stuff")
  results = chain({
    'input_documents': docs,
    "question": update.message.text
  },
                  return_only_outputs=True)
  text = results['output_text']
  await context.bot.send_message(chat_id=update.effective_chat.id, text=text)


if __name__ == "__main__":
  application = ApplicationBuilder().token(
    os.getenv('TELEGRAM_BOT_TOKEN')).build()
  application.add_handler(CommandHandler('start', start))
  application.add_handler(CommandHandler('load', load))
  application.add_handler(CommandHandler('query', query))
  application.run_polling()
