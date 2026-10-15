---
layout: page
permalink: /paper/
title: paper archive
description: Stable full-text records of research papers and manuscripts.
nav: false
---

{% assign archive_papers = site.papers | sort: 'date' | reverse %}

<div class="paper-archive-intro">
  <p>
    Each record has a permanent local identifier, complete citation metadata, and an immutable archived PDF.
    External identifiers such as DOI and arXiv are added when available.
  </p>
</div>

<div class="paper-archive-list">
  {% for paper in archive_papers %}
    <article class="paper-archive-item">
      <div class="paper-archive-meta">
        <span>{{ paper.status }}</span>
        <span>{{ paper.date | date: '%Y-%m-%d' }}</span>
        <span>{{ paper.archive_id }}</span>
      </div>
      <h2><a href="{{ paper.url | relative_url }}">{{ paper.title }}</a></h2>
      <p class="paper-archive-authors">
        {% for author in paper.authors %}
          {{ author.name }}{% unless forloop.last %}, {% endunless %}
        {% endfor %}
      </p>
      <p>{{ paper.abstract | truncatewords: 55 }}</p>
      <div class="paper-actions">
        <a class="btn btn-sm btn-primary" href="{{ paper.url | relative_url }}">Abstract</a>
        <a class="btn btn-sm btn-outline-primary" href="{{ paper.pdf | relative_url }}">PDF</a>
      </div>
    </article>
  {% else %}
    <p>No archived papers yet.</p>
  {% endfor %}
</div>
