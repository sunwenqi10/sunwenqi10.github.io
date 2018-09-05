---
layout: post
title: "Windows下使用github+Jekyll搭建个人博客"
tags: [Jekyll, Github]
date: 2018-09-03
---

主要流程参考[Jonathan McGlone](http://jmcglone.com/guides/github-pages/).
1. 按照上述链接搭建起一个简易框架（根据自己需要适当修改），例如[sunwenqi10.github.io](https://github.com/sunwenqi10/sunwenqi10.github.io)。
2. 在本地电脑上下载[Github Deaktop](https://desktop.github.com/)以及[Jekyll](https://jekyllrb.com/docs/installation/)。Windows下安装Jekyll时Ruby路径最好不要有中文和空格，否则可能报错，推荐安装包默认设置。
3. 将github的仓库username.github.io克隆到本地的文件夹，在命令窗口下进入该文件夹，输入**jekyll server**开启本地编辑和调试。
4. 下载[Atom](https://atom.io/)编辑器在本地编辑markdown或html文件，本地调试好之后将改动push到github的仓库中。
