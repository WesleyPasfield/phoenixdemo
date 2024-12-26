#!/bin/bash

STACK_NAME="llm-eval"
TEMPLATE_FILE="template.yaml"
PARAMS_FILE="parameters.json"

aws cloudformation create-stack \
 --stack-name $STACK_NAME \
 --template-body file://infrastructure/$TEMPLATE_FILE \
 --parameters file://infrastructure/$PARAMS_FILE \
 --capabilities CAPABILITY_IAM

aws cloudformation wait stack-create-complete \
 --stack-name $STACK_NAME

echo "Stack outputs:"
aws cloudformation describe-stacks \
 --stack-name $STACK_NAME \
 --query 'Stacks[0].Outputs'