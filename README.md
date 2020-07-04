### Summarization with clustering the DistillBERT text representations

The parts of the code is from the repo `https://github.com/dmmiller612/bert-extractive-summarizer` modified to be used with streamlit

- It does not make use of coreference resolution

It is based on the paper `https://arxiv.org/abs/1906.04165`


### building docker container

```bash
docker built . -t streamlit-sum
docker run -p 80:8501 streamlit-sum:latest
```

should be accessible in 
 `localhost:80`
or 
 `<Externel IP>:80` after making sure no firewall restrictions on your port 80
