---
layout: page
permalink: /people/
title: people
description: members of the lab, including students and interns
nav: true
nav_order: 7

# Add members to the matching list below. Keep sort_name in pinyin order so the
# cards stay alphabetized. Put profile photos in assets/img/ (for example,
# assets/img/students/alice.jpg) and use that relative path as image.
students:
  # undergraduate:
  phd:
    - name: Zicai Cui
      image: students/zicai-cui.jpg
      email: zicaicui@sjtu.edu.cn
      sort_name: Cui Zicai
      affiliation: Shanghai Jiao Tong University
      research: Intelligent Agents
    - name: Zihan Guo
      image: students/zihan-guo.jpg
      email: guozh29@mail2.sysu.edu.cn
      sort_name: Guo Zihan
      affiliation: Chuangzhi College
      research: Intelligent Agents
      co_supervised_by: Prof. Weinan Zhang
    - name: Zhi Han
      image: students/zhi-han.jpg
      email: hanzhi_sjtu669@sjtu.edu.cn
      sort_name: Han Zhi
      affiliation: Shanghai Jiao Tong University
      research: Intelligent Agents
    - name: Xiaohan Mao
      image: students/xiaohan-mao.jpg
      email: mxh1999@sjtu.edu.cn
      sort_name: Mao Xiaohan
      affiliation: Shanghai AI Laboratory · Joint Ph.D. student
      research: Embodied AI
    - name: Junru Song
      image: students/junru-song.jpg
      email: woodysjr@foxmail.com
      sort_name: Song Junru
      affiliation: Shanghai Jiao Tong University
      research: Embodied Agents
      co_supervised_by: Prof. Ying Wen
    - name: Yang Tang
      image: students/yang-tang.jpg
      email: 2629535324@sjtu.edu.cn
      sort_name: Tang Yang
      affiliation: Shanghai Jiao Tong University
      research: Embodied Agents
      co_supervised_by: Prof. Weinan Zhang
    - name: Chenyang Wan
      image: students/chenyang-wan.jpg
      email: cywan@zju.edu.cn
      homepage: https://bryce-wan.github.io/
      sort_name: Wan Chenyang
      affiliation: Shanghai AI Laboratory · Joint Ph.D. student
      research: Embodied AI
    - name: Wenhao Zhang
      image: students/wenhao-zhang.jpg
      email: wenhao_zhang@sjtu.edu.cn
      homepage: https://github.com/Bluixe
      sort_name: Zhang Wenhao
      affiliation: Shanghai Jiao Tong University
      research: In-context Reinforcement Learning
      co_supervised_by: Prof. Ying Wen
  master:
    - name: Fanxin Shen
      image: students/fanxin-shen.jpg
      email: shenzju2022@163.com
      sort_name: Shen Fanxin
      affiliation: Shanghai Jiao Tong University · Master's student, Class of 2026
      research: Intelligent Agents
  intern:
    - name: Wenbo Fei
      image: students/wenbo-fei.jpg
      email: fei001@sjtu.edu.cn
      sort_name: Fei Wenbo
      affiliation: Shanghai Jiao Tong University · Undergraduate, Class of 2024
      research: Embodied AI
    - name: Yue Xin
      image: students/yue-xin.jpg
      email: vinkda419@gmail.com
      sort_name: Xin Yue
      affiliation: Beihang University · Graduate student, Class of 2024
      research: Dexterous Manipulation and World Models
    - name: Jiaqi Xu
      image: students/jiaqi-xu.jpg
      email: dlrbdlmltzzn@sjtu.edu.cn
      sort_name: Xu Jiaqi
      affiliation: Shanghai Jiao Tong University · Undergraduate, Class of 2024
      research: Physical Agents
    - name: Chenxi Zeng
      image: students/chenxi-zeng.jpg
      email: zengcx2423@mails.jlu.edu.cn
      homepage: https://chenxizeng930.github.io
      sort_name: Zeng Chenxi
      affiliation: Jilin University · Undergraduate, Class of 2023
      research: Physical Agents
---

<div class="students">
  <p class="students-intro">
    Our students and interns work across embodied AI, multi-agent systems, robotics, and AI for science.
    Select a name to visit a personal homepage when one is available.
  </p>

  {% assign student_groups = "undergraduate|master|phd|intern" | split: "|" %}
  {% for group in student_groups %}
    {% assign members = page.students[group] %}
    {% if members and members.size > 0 %}
      <section class="student-group" id="{{ group }}">
        {% case group %}
          {% when "undergraduate" %}
            <h2>Undergraduate</h2>
          {% when "master" %}
            <h2>Master</h2>
          {% when "phd" %}
            <h2>Ph.D.</h2>
          {% when "intern" %}
            <h2>Intern</h2>
        {% endcase %}

        {% assign sorted_members = members | sort: "sort_name" %}
        <div class="student-grid">
          {% for student in sorted_members %}
            <article class="student-card">
              {% assign student_image = student.image | default: "no_photo.png" | prepend: "assets/img/" | relative_url %}
              <img class="student-card-image" src="{{ student_image }}" alt="{{ student.name }}" loading="lazy">
              <div class="student-card-body">
                {% if student.homepage %}
                  <h3><a href="{{ student.homepage }}" target="_blank" rel="noopener">{{ student.name }}</a></h3>
                {% else %}
                  <h3>{{ student.name }}</h3>
                {% endif %}
                {% if student.affiliation %}
                  <p class="student-affiliation">{{ student.affiliation }}</p>
                {% endif %}
                {% if student.email %}
                  <p class="student-email" style="overflow-wrap: anywhere; font-size: 0.82rem; margin: 0.5rem 0 0;"><a href="mailto:{{ student.email }}">{{ student.email }}</a></p>
                {% endif %}
                {% if student.research %}
                  <p class="student-research"><span class="student-label">Research</span> {{ student.research }}</p>
                {% endif %}
                {% if student.homepage %}
                  <a class="student-homepage" href="{{ student.homepage }}" target="_blank" rel="noopener">Homepage <span aria-hidden="true">↗</span></a>
                {% endif %}
                {% if student.co_supervised_by %}
                  <p class="student-cosupervisor">Co-supervised by {{ student.co_supervised_by }}</p>
                {% endif %}
              </div>
            </article>
          {% endfor %}
        </div>
      </section>
    {% endif %}
  {% endfor %}
</div>

<!--
Example entry:
- name: Student Name
  sort_name: Student Name
  affiliation: University · year/program
  image: students/student-name.jpg
  email: student@example.com
  research: Embodied AI and multi-agent learning
  homepage: https://example.com
  co_supervised_by: Prof. Co-supervisor
-->
