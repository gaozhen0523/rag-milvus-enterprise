resource "aws_iam_openid_connect_provider" "github" {
  url = "https://token.actions.githubusercontent.com"

  client_id_list = [
    "sts.amazonaws.com"
  ]

  thumbprint_list = [
    "6938fd4d98bab03faadb97b34396831e3780aea1"
  ]
}

resource "aws_iam_role" "github_actions_role" {
  name = "rag-github-actions-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17",
    Statement = [
      {
        Effect = "Allow",
        Principal = {
          Federated = aws_iam_openid_connect_provider.github.arn
        },
        Action = "sts:AssumeRoleWithWebIdentity",
        Condition = {
          StringEquals = {
            # 只允许你的 repo + main 分支
            "token.actions.githubusercontent.com:sub" = "repo:gaozhen0523/rag-milvus-enterprise:ref:refs/heads/main"
          }
        }
      }
    ]
  })
}

# 你的 GitHub Actions 需要 ECR + ECS + CloudWatch Logs + IAM PassRole + EC2 Describe（用于 pull subnet/SG）
resource "aws_iam_role_policy" "github_actions_policy" {
  role = aws_iam_role.github_actions_role.id

  policy = jsonencode({
    Version = "2012-10-17",
    Statement = [
      {
        Effect = "Allow",
        Action = [
          "ecr:*",
          "ecs:*",
          "iam:PassRole",
          "logs:*",
          "ssm:GetParameter",
          "sts:GetCallerIdentity",
          "ec2:Describe*",

          # 让 CI 能读 tfstate
          "s3:GetObject",
          "s3:ListBucket",

          # ECS Task Execution Role / Task Role creation needs:
          "iam:CreateRole",
          "iam:DeleteRole",
          "iam:GetRole",
          "iam:PassRole",
          "iam:PutRolePolicy",
          "iam:AttachRolePolicy",
          "iam:DetachRolePolicy",
          "iam:UpdateAssumeRolePolicy",
          "iam:ListAttachedRolePolicies",
          "iam:CreatePolicy",
          "iam:GetPolicy",
          "iam:ListPolicyVersions",
          "iam:CreatePolicyVersion",
          "iam:DeletePolicyVersion",
          "iam:TagRole",
          "iam:CreateOpenIDConnectProvider",
        ],
        Resource = "*"
      }
    ]
  })
}

output "github_actions_role_arn" {
  value = aws_iam_role.github_actions_role.arn
}
