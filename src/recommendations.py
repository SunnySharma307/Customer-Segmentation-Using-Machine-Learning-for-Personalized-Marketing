"""
recommendations.py
Actionable marketing recommendation engine driven by Customer Segment and Response Probability.
"""

from typing import Dict, Any, Optional


def generate_recommendation(
    cluster_name: str,
    response_probability: float,
    is_responsive: bool,
    customer_data: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Generates personalized marketing recommendations based on the customer's
    segment characteristics and predicted campaign responsiveness.
    """
    c_name = cluster_name.lower()
    prob_pct = round(response_probability * 100, 1)

    # Detect high spending affinities if customer data provided
    top_category = "Curated Products"
    preferred_channel = "Omnichannel"
    if customer_data:
        spends = {
            "Premium Wines": customer_data.get("MntWines", 0),
            "Gourmet Meats": customer_data.get("MntMeatProducts", 0),
            "Confectionery & Sweets": customer_data.get("MntSweetProducts", 0),
            "Gold Luxury Items": customer_data.get("MntGoldProds", 0),
            "Fresh Seafood": customer_data.get("MntFishProducts", 0),
        }
        if any(v > 0 for v in spends.values()):
            top_category = max(spends, key=spends.get)

        web_p = customer_data.get("NumWebPurchases", 0)
        store_p = customer_data.get("NumStorePurchases", 0)
        catalog_p = customer_data.get("NumCatalogPurchases", 0)
        channels = {"Digital Web / App": web_p, "Physical In-Store": store_p, "Direct Mail Catalog": catalog_p}
        if any(v > 0 for v in channels.values()):
            preferred_channel = max(channels, key=channels.get)

    # Segment + Response Strategy Matrix
    if "high-value" in c_name or "elite" in c_name:
        if is_responsive:
            strategy = "VIP Loyalty & Exclusive Privileges"
            offer_type = f"Exclusive Sommelier / Chef Reserve bundle for {top_category}"
            channel = "Direct Mail Luxury Catalog & Dedicated Account Manager"
            tone = "Prestigious, personalized, white-glove, recognizing high status"
            expected_roi = "Very High (High Margin & High Conversion)"
            action_items = [
                "Invite to exclusive private tasting or private shopping salon.",
                f"Offer complimentary high-tier gift with purchase on {top_category}.",
                "Enroll in invitation-only Platinum tier with priority dispatch.",
            ]
        else:
            strategy = "Prestige Re-Engagement & Appreciation"
            offer_type = "Zero-obligation complimentary VIP anniversary gift box"
            channel = "Personalized Letter from Head Sommelier + Discreet SMS"
            tone = "Warm appreciation, respectful of time, high-touch without aggressive sales"
            expected_roi = "High (Prevents high-value customer churn)"
            action_items = [
                "Send executive thank-you note recognizing customer loyalty.",
                "Provide bespoke consultation or concierge booking for upcoming seasonal stock.",
                "Offer flexible concierge delivery options.",
            ]

    elif "deal" in c_name or "discount" in c_name or "budget" in c_name:
        if is_responsive:
            strategy = "High-Urgency Flash Sale & Bundle Incentives"
            offer_type = f"Buy-2-Get-1 Free or 25% off coupon on {top_category}"
            channel = "Targeted Email Blast & In-Store Coupon Push"
            tone = "Exciting, value-focused, time-sensitive urgency ('48 Hours Only')"
            expected_roi = "Moderate to High (High Volume, Elastic Demand)"
            action_items = [
                "Highlight clear price savings and percentage discounts prominently.",
                "Trigger automated countdown emails for flash weekend deals.",
                "Reward deal loyalty with points bonus on multi-item purchases.",
            ]
        else:
            strategy = "Low-Cost Threshold Couponing"
            offer_type = "Free delivery on orders over ₹500 + ₹100 off next purchase"
            channel = "Automated Email Notification"
            tone = "Straightforward savings, accessible entry price point"
            expected_roi = "Moderate (Requires strict margin control)"
            action_items = [
                "Avoid expensive direct mail; utilize automated low-cost digital channels.",
                "Test low-friction entry offers with immediate redeemable value.",
                "Bundle clearance items to preserve inventory turnover.",
            ]

    elif "digital" in c_name or "web" in c_name:
        if is_responsive:
            strategy = "Personalized Digital Experience & Dynamic Cross-Sell"
            offer_type = f"Interactive Web recommendation engine bundle for {top_category}"
            channel = "Mobile Push, Personalized Homepage Banner & Email Newsletter"
            tone = "Modern, seamless, frictionless, digital-first"
            expected_roi = "High (Low acquisition cost per click)"
            action_items = [
                "Feature 1-click reorder banners tailored to past browsing behavior.",
                "Offer app-exclusive preview of new seasonal arrivals.",
                "Use abandoned-cart reminders with a dynamic 10% instant rebate.",
            ]
        else:
            strategy = "Digital Engagement Nudge & UX Re-activation"
            offer_type = "500 bonus loyalty points on next web checkout"
            channel = "Email Drip Campaign & Web Retargeting"
            tone = "Helpful, user-friendly, highlighting convenience features"
            expected_roi = "Moderate (Low digital marginal cost)"
            action_items = [
                "Send survey asking for digital preferences or wishlist updates.",
                "Highlight improved web checkout speed and mobile payment ease.",
                "Provide digital recipe cards or pairing guides featuring store products.",
            ]

    else:  # Occasional / Low Engagement
        if is_responsive:
            strategy = "Re-Activation & First Re-Purchase Incentive"
            offer_type = f"₹500 welcome-back voucher valid on any {top_category}"
            channel = "Email with fallback to SMS"
            tone = "Friendly, welcoming, 'We miss you' re-connection"
            expected_roi = "Moderate (Re-engages dormant revenue)"
            action_items = [
                "Present simple, curated top-3 bestsellers to eliminate choice fatigue.",
                "Offer no-minimum-purchase free shipping for their return purchase.",
                "Follow up with a post-purchase satisfaction check-in.",
            ]
        else:
            strategy = "Low-Frequency Drip & Retention Monitoring"
            offer_type = "Seasonal catalog summary or general brand newsletter"
            channel = "Low-frequency email list"
            tone = "Informative, non-intrusive"
            expected_roi = "Low to Neutral (Preventing marketing waste)"
            action_items = [
                "Do not spend on high-cost outreach (direct mail or calls).",
                "Keep customer in passive quarterly updates to maintain brand recall.",
                "Monitor for organic web visits before investing promotional budget.",
            ]

    return {
        "segment": cluster_name,
        "response_probability_pct": prob_pct,
        "is_responsive": is_responsive,
        "predicted_label": "Likely to Respond" if is_responsive else "Unlikely to Respond",
        "primary_strategy": strategy,
        "campaign_offer": offer_type,
        "recommended_channel": preferred_channel or channel,
        "messaging_tone": tone,
        "expected_roi": expected_roi,
        "action_items": action_items,
        "affinity_category": top_category,
    }
