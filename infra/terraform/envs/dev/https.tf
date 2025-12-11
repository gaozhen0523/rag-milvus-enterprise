#############################
# Route53 Hosted Zone
#############################

data "aws_route53_zone" "root" {
  name         = var.root_domain
  private_zone = false
}

#############################
# ACM certificate for rag-api-gateway
#############################

resource "aws_acm_certificate" "rag_api" {
  domain_name       = "${var.rag_api_subdomain}.${var.root_domain}"
  validation_method = "DNS"

  lifecycle {
    create_before_destroy = true
  }
}

resource "aws_route53_record" "rag_api_cert_validation" {
  for_each = {
    for dvo in aws_acm_certificate.rag_api.domain_validation_options : dvo.domain_name => {
      name   = dvo.resource_record_name
      type   = dvo.resource_record_type
      record = dvo.resource_record_value
    }
  }

  zone_id = data.aws_route53_zone.root.zone_id
  name    = each.value.name
  type    = each.value.type
  ttl     = 60
  records = [each.value.record]
}

resource "aws_acm_certificate_validation" "rag_api" {
  certificate_arn         = aws_acm_certificate.rag_api.arn
  validation_record_fqdns = [for r in aws_route53_record.rag_api_cert_validation : r.fqdn]
}

#############################
# ALB data source (replace module name!)
#############################

data "aws_lb" "rag_api" {
  # TODO: 把 rag_api_gateway_service 换成你实际 env 里的 module 名字
  arn = module.rag_api_gateway_service.alb_arn
}

#############################
# Route53 A record -> rag-api-gateway ALB
#############################

resource "aws_route53_record" "rag_api_alias" {
  zone_id = data.aws_route53_zone.root.zone_id
  name    = "${var.rag_api_subdomain}.${var.root_domain}"
  type    = "A"

  alias {
    name                   = data.aws_lb.rag_api.dns_name
    zone_id                = data.aws_lb.rag_api.zone_id
    evaluate_target_health = true
  }
}

#############################
# HTTPS Listener for rag-api-gateway ALB
#############################

resource "aws_lb_listener" "rag_api_https" {
  load_balancer_arn = data.aws_lb.rag_api.arn
  port              = 443
  protocol          = "HTTPS"
  ssl_policy        = "ELBSecurityPolicy-2016-08"
  certificate_arn   = aws_acm_certificate_validation.rag_api.certificate_arn

  default_action {
    type             = "forward"
    target_group_arn = module.rag_api_gateway_service.target_group_arn # 同样注意 module 名字
  }
}
