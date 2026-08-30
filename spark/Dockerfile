FROM apache/spark:3.5.6

USER root

RUN mkdir -p /opt/spark/jobs

ADD https://repo1.maven.org/maven2/org/apache/hadoop/hadoop-aws/3.3.4/hadoop-aws-3.3.4.jar \
    /opt/spark/jars/hadoop-aws-3.3.4.jar

ADD https://repo1.maven.org/maven2/com/amazonaws/aws-java-sdk-bundle/1.12.262/aws-java-sdk-bundle-1.12.262.jar \
    /opt/spark/jars/aws-java-sdk-bundle-1.12.262.jar

RUN chmod 644 /opt/spark/jars/hadoop-aws-3.3.4.jar \
    /opt/spark/jars/aws-java-sdk-bundle-1.12.262.jar

RUN chmod 755 /opt/spark/jars

USER 185