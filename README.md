# Generative model

Generative AI, Efficient AI, Uncertainty Quantification, Probabilistic ML에 관한 공부 기록과 논문 리뷰를 위한 블로그입니다.

- 게시 주소: https://shkim99stat.github.io
- 기반: [Chirpy Starter](https://github.com/cotes2020/chirpy-starter)
- 언어: ko-KR
- 시간대: Asia/Seoul

## 글 작성

1. `templates/`에서 필요한 양식을 복사합니다.
2. `_posts/YYYY-MM-DD-english-slug.md`로 저장합니다.
3. 제목, 설명, 날짜, 태그와 본문을 작성합니다.
4. 공개할 글은 `published: true`로 바꿉니다.
5. main에 변경을 반영하면 GitHub Actions가 빌드하고 게시합니다.

| 글 유형 | 카테고리 | 양식 |
|---|---|---|
| 공부 기록 | 공부 기록 | templates/study-note.md |
| 논문 리뷰 | 논문 리뷰 | templates/paper-review.md |
| 구현 및 재현 | 프로젝트 | templates/project.md |

주제 태그: `generative-ai`, `efficient-ai`, `uncertainty-quantification`, `probabilistic-ml`.

수식이 필요한 글은 `math: true`, 홈에서 강조할 글은 `pin: true`로 설정합니다.
프로젝트 페이지는 `프로젝트` 카테고리 글을 자동으로 모읍니다.

양식 폴더는 빌드에서 제외됩니다. `published: false`인 글도 사이트에는 게시되지 않지만, 공개 저장소의 원본 파일은 누구나 GitHub에서 볼 수 있습니다.

## 처음 게시하기

저장소 Settings → Pages → Build and deployment → Source를 **GitHub Actions**로 선택합니다.
main에 반영하거나 Actions → Build and Deploy Blog → Run workflow로 배포합니다.
PR에서는 사이트 빌드와 내부 링크 검사만 실행합니다.

## 로컬 확인

이 저장소를 clone한 뒤 Ruby 3.4와 Bundler가 있는 환경에서 실행합니다.

```text
bundle install
bundle exec jekyll serve
```

## 설정 파일

- `_config.yml`: 제목, 언어, 시간대, 기본 주소
- `_tabs/about.md`: 소개
- `_tabs/projects.md`: 프로젝트 모음
- `_data/contact.yml`: GitHub 및 RSS 링크
- `assets/css/jekyll-theme-chirpy.scss`: 한글 가독성 조정
- `.github/workflows/pages-deploy.yml`: 빌드, 링크 검사, Pages 배포

## 공식 문서

- [GitHub Pages](https://docs.github.com/en/pages/getting-started-with-github-pages/what-is-github-pages)
- [Chirpy 설치와 배포](https://chirpy.cotes.page/posts/getting-started/)
- [Chirpy 글 작성](https://chirpy.cotes.page/posts/write-a-new-post/)

원본 테마와 Starter의 MIT 라이선스는 LICENSE에 보존합니다.
