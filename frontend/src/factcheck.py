import wikipedia

def wiki_evidence(query, max_results=3):
    try:
        results = wikipedia.search(query, results=max_results)
        summaries = []
        for r in results:
            try:
                summaries.append({'title': r, 'summary': wikipedia.summary(r, sentences=1)})
            except:
                summaries.append({'title': r, 'summary': ''})
        return summaries
    except Exception as e:
        return []
